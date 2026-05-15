from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests
from loguru import logger
from rich import print
from tqdm import tqdm

CACHE_DIR = Path(__file__).parent.parent / "cache"
DEFAULT_SAMPLE_SEED = 42

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with open(tmp_path, "w") as f:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}) + "\n")
    os.replace(tmp_path, out_path)

    with open(out_path) as f:
        num_samples = sum(1 for _ in f)
    print(f"[cached] {out_path}  ({num_samples} samples)")
    return out_path


def load_and_process_dataset(data_name: str) -> list[dict]:
    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    path = CACHE_DIR / f"{data_name}.jsonl"
    if not path.exists():
        _prepare_dataset(data_name)

    with open(path) as f:
        return [json.loads(line) for line in f]


def _limit_dataset(dataset: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    if max_samples is None or len(dataset) <= max_samples:
        return list(dataset)
    selected = list(dataset)
    random.Random(seed).shuffle(selected)
    return selected[:max_samples]


def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _make_decode_metrics(
    num_output_tokens: int,
    generation_tps: float,
    acceptance_lengths: list[int],
    profile: dict[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_output_tokens=num_output_tokens,
        time_per_output_token=1.0 / generation_tps if generation_tps > 0 else float("inf"),
        acceptance_lengths=acceptance_lengths,
        profile=profile,
    )


def _sum_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for profile in profiles:
        for key, value in profile.items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _print_profile_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_profiles = [r[1].profile for r in responses if r[1].profile]
    if baseline_profiles:
        baseline_totals = _sum_profiles(baseline_profiles)
        baseline_tokens = sum(r[1].num_output_tokens for r in responses)
        baseline_s = baseline_totals.get("baseline_s", 0.0)
        print("\nBaseline profile totals:")
        print(f"  baseline_s: {baseline_s:.3f}s ({baseline_s / max(baseline_tokens, 1) * 1000:.2f} ms/output tok)")
        print(f"  emitted tokens: {baseline_tokens}")

    profiles = [r[block_size].profile for r in responses if r[block_size].profile]
    if not profiles:
        return

    totals = _sum_profiles(profiles)
    num_tokens = sum(r[block_size].num_output_tokens for r in responses)
    num_steps = max(int(totals.get("steps", len(profiles))), 1)
    print("\nDFlash profile totals:")
    for key in [
        "prefill_s",
        "cache_checkpoint_s",
        "draft_s",
        "verify_s",
        "cache_trim_s",
        "cache_replay_s",
    ]:
        value = totals.get(key, 0.0)
        print(f"  {key}: {value:.3f}s ({value / max(num_tokens, 1) * 1000:.2f} ms/output tok)")
    print(f"  profile steps: {num_steps}")
    print(f"  emitted tokens: {num_tokens}")
    print(f"  avg accepted/profile step: {totals.get('accepted', 0.0) / num_steps:.2f}")
    print(f"  avg block size/profile step: {totals.get('block_size', 0.0) / num_steps:.2f}")
    print(f"  target cache trimmable: {bool(round(totals.get('target_cache_trimmable', 0.0) / num_steps))}")
    print(f"  block verify: {bool(round(totals.get('block_verify', 0.0) / num_steps))}")


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = np.mean([r[1].time_per_output_token for r in responses])
    dflash_tpot = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    mean_accept = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {mean_accept:.2f}")

    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    _print_profile_summary(responses, block_size)


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _token_window(tokens: list[int], center: int, radius: int = 6) -> list[int]:
    start = max(center - radius, 0)
    end = min(center + radius + 1, len(tokens))
    return tokens[start:end]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dist_init(torch_dist) -> None:
    if "RANK" not in os.environ:
        warnings.warn("RANK not set. Skipping distributed initialization.")
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _dist_rank() -> int:
    return _env_int("RANK", 0)


def _dist_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(torch_dist, obj: Any, dst: int = 0):
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


_TRANSFORMERS_SUPPORTED_PATTERN = re.compile(r"qwen3(?!\.5)[\w-]*|llama.*3\.1.*8b.*instruct", re.IGNORECASE)


def _check_transformers_model(model_name: str) -> None:
    if not _TRANSFORMERS_SUPPORTED_PATTERN.search(model_name):
        raise ValueError(
            f"Transformers backend does not support '{model_name}'. "
            f"Only Qwen3 series and LLaMA-3.1-8B-Instruct are supported. "
            f"Use --backend sglang or --backend vllm for other models."
        )


def _get_transformers_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash_attn not installed. Falling back to torch.sdpa. Speedup will be lower. "
            "For optimal speedup in Transformers backend, please install: "
            "pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


def _run_transformers(args: argparse.Namespace) -> None:
    import torch
    from torch import distributed as torch_dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model import DFlashDraftModel, dflash_generate

    _check_transformers_model(args.model)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _dist_init(torch_dist)
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")
    attn_impl = _get_transformers_attn_impl()

    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)

    dataset = _limit_dataset(dataset, args.max_samples)

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    for idx in tqdm(indices, disable=not _dist_is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if _dist_size() > 1:
        responses = _dist_gather(torch_dist, responses, dst=0)
        if not _dist_is_main():
            return
        responses = list(chain(*responses))

    _print_decode_summary(responses, block_size)


def _send_sglang(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def _send_vllm(
    base_url: str,
    text: str,
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
    enable_thinking: bool = False,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _run_mlx(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import stream_generate as stream_generate_baseline
    from mlx_lm.sample_utils import make_sampler

    from .model_mlx_clean import load, load_draft, stream_generate

    if args.check_equivalence and args.temperature != 0.0:
        raise ValueError("--check-equivalence requires greedy decoding with --temperature 0.0")

    sampler = make_sampler(temp=args.temperature)

    logger.info(f"Loading target: {args.model}")
    model, tokenizer = load(args.model)
    logger.info(f"Loading draft: {args.draft_model}")
    draft = load_draft(args.draft_model)
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples, args.sample_seed)

    warmup_prompt = tokenizer.encode("Hi")
    list(stream_generate_baseline(model, tokenizer, warmup_prompt, 3, sampler=sampler))
    list(stream_generate(
        model,
        draft,
        tokenizer,
        warmup_prompt,
        block_size,
        3,
        sampler=sampler,
        profile=args.profile,
    ))

    responses = []
    equivalence_failures = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for turn_idx, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(tokenizer, messages, args.enable_thinking)

            response = {}

            tokens_bl, tps_bl = [], 0
            baseline_start = time.perf_counter()
            if args.profile:
                mx.synchronize()
                baseline_start = time.perf_counter()
            for r in stream_generate_baseline(model, tokenizer, prompt, args.max_new_tokens, sampler=sampler):
                tokens_bl.append(r.token)
                tps_bl = r.generation_tps
            baseline_profile = None
            if args.profile:
                mx.synchronize()
                baseline_profile = {"baseline_s": time.perf_counter() - baseline_start}
            response[1] = _make_decode_metrics(len(tokens_bl), tps_bl, [1], baseline_profile)

            tokens_df, accs, tps_df, profiles = [], [], 0, []
            for r in stream_generate(
                model,
                draft,
                tokenizer,
                prompt,
                block_size,
                args.max_new_tokens,
                sampler=sampler,
                profile=args.profile,
            ):
                if r.tokens:
                    tokens_df.extend(r.tokens)
                    accs.append(r.accepted)
                if r.profile:
                    profiles.append(r.profile)
                tps_df = r.generation_tps
            response[block_size] = _make_decode_metrics(
                len(tokens_df),
                tps_df,
                accs,
                _sum_profiles(profiles) if profiles else None,
            )

            if args.check_equivalence:
                mismatch = _first_mismatch(tokens_bl, tokens_df)
                if mismatch is not None:
                    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
                    equivalence_failures.append({
                        "sample": idx,
                        "turn": turn_idx,
                        "index": mismatch,
                        "prompt_sha1": prompt_hash,
                        "prompt_chars": len(prompt),
                        "prompt_preview": user_content[:160].replace("\n", "\\n"),
                        "baseline_token": tokens_bl[mismatch] if mismatch < len(tokens_bl) else None,
                        "dflash_token": tokens_df[mismatch] if mismatch < len(tokens_df) else None,
                        "baseline_window": _token_window(tokens_bl, mismatch),
                        "dflash_window": _token_window(tokens_df, mismatch),
                        "baseline_len": len(tokens_bl),
                        "dflash_len": len(tokens_df),
                    })

            output_text = tokenizer.decode(tokens_df)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)
    if args.check_equivalence:
        checked = sum(len(item["turns"]) for item in dataset)
        if equivalence_failures:
            print(f"Equivalence check: FAILED ({len(equivalence_failures)}/{checked} turns)")
            for failure in equivalence_failures[:5]:
                print(f"  {failure}")
            raise AssertionError("DFlash output diverged from target-only greedy output")
        print(f"Equivalence check: passed ({checked} turns)")


def _run_server(args: argparse.Namespace) -> None:
    is_vllm = args.backend == "vllm"
    dataset = load_and_process_dataset(args.dataset)
    tokenizer = None
    if not is_vllm:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    num_prompts = args.num_prompts + args.concurrency
    prompts: list[str] = []
    for i in range(num_prompts):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        if is_vllm:
            prompts.append(user_content)
        else:
            prompts.append(_apply_chat_template(
                tokenizer,
                [{"role": "user", "content": user_content}],
                args.enable_thinking,
            ))

    def send_one(prompt: str) -> dict:
        if is_vllm:
            return _send_vllm(
                args.base_url,
                prompt,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                timeout_s=args.timeout_s,
                enable_thinking=args.enable_thinking,
            )
        return _send_sglang(
            args.base_url,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
        )

    if not is_vllm:
        try:
            requests.get(args.base_url + "/flush_cache", timeout=60).raise_for_status()
        except Exception:
            print("Warning: /flush_cache failed. Continuing.")

    bs = max(args.concurrency, 1)
    if len(prompts) > bs:
        print(f"[warmup] {bs} requests ...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(send_one, prompts[:bs]))
        prompts = prompts[bs:]

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={args.concurrency} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(send_one, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            if is_vllm:
                usage = out.get("usage", {})
                total_tokens += int(usage.get("completion_tokens", 0))
            else:
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark")
    parser.add_argument("--backend", choices=["transformers", "sglang", "vllm", "mlx"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Seed for deterministic benchmark sample selection",
    )

    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--profile", action="store_true", help="Enable synchronized MLX DFlash profiling")
    parser.add_argument("--check-equivalence", action="store_true", help="Assert MLX DFlash matches target-only greedy output")
    args = parser.parse_args()

    assert not (args.enable_thinking and any(x in args.model.lower() for x in ["qwen3-4b", "qwen3-8b"])), (
        "DFlash draft models for Qwen3-4B and Qwen3-8B were not trained with thinking traces. "
        "Using --enable-thinking will lead to suboptimal performance."
    )

    if args.backend == "transformers":
        if args.draft_model is None:
            parser.error("--draft-model is required for transformers backend")
        _run_transformers(args)
    elif args.backend == "mlx":
        if args.draft_model is None:
            parser.error("--draft-model is required for mlx backend")
        _run_mlx(args)
    else:
        _run_server(args)


if __name__ == "__main__":
    main()
