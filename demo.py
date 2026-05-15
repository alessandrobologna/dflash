from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-4B-DFlash"


@dataclass
class RunStats:
    name: str
    text: str
    tokens: int
    elapsed_s: float
    generation_tps: float
    peak_memory_gb: float
    acceptance_lengths: list[int]

    @property
    def measured_tps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.tokens / self.elapsed_s


def _load_runtime() -> SimpleNamespace:
    import mlx.core as mx
    from mlx_lm import stream_generate as stream_generate_baseline
    from mlx_lm.sample_utils import make_sampler

    from dflash.model_mlx_clean import load, load_draft, stream_generate

    return SimpleNamespace(
        mx=mx,
        stream_generate_baseline=stream_generate_baseline,
        make_sampler=make_sampler,
        load=load,
        load_draft=load_draft,
        stream_generate=stream_generate,
    )


def _make_prompt(tokenizer: Any, user_prompt: str, *, enable_thinking: bool, raw_prompt: bool) -> str:
    if raw_prompt:
        return user_prompt

    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _run_target(
    runtime: SimpleNamespace,
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    sampler: Any,
    stream: bool,
) -> RunStats:
    if stream:
        print("\n[target-only]\n", flush=True)

    text_parts: list[str] = []
    tokens = 0
    last_tps = 0.0
    peak_memory = 0.0

    runtime.mx.synchronize()
    start = time.perf_counter()
    for response in runtime.stream_generate_baseline(
        model,
        tokenizer,
        prompt,
        max_tokens,
        sampler=sampler,
    ):
        segment = getattr(response, "text", "")
        if stream and segment:
            print(segment, end="", flush=True)
        text_parts.append(segment)
        tokens += 1
        last_tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
        peak_memory = max(peak_memory, float(getattr(response, "peak_memory", 0.0) or 0.0))
    runtime.mx.synchronize()
    elapsed = time.perf_counter() - start

    if stream:
        print("\n", flush=True)

    return RunStats(
        name="target-only",
        text="".join(text_parts),
        tokens=tokens,
        elapsed_s=elapsed,
        generation_tps=last_tps,
        peak_memory_gb=peak_memory,
        acceptance_lengths=[],
    )


def _run_dflash(
    runtime: SimpleNamespace,
    model: Any,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    *,
    block_size: int,
    max_tokens: int,
    sampler: Any,
    stream: bool,
) -> RunStats:
    if stream:
        print("\n[dflash]\n", flush=True)

    text_parts: list[str] = []
    tokens = 0
    last_tps = 0.0
    peak_memory = 0.0
    acceptance_lengths: list[int] = []

    runtime.mx.synchronize()
    start = time.perf_counter()
    for response in runtime.stream_generate(
        model,
        draft,
        tokenizer,
        prompt,
        block_size=block_size,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        segment = response.text
        if stream and segment:
            print(segment, end="", flush=True)
        text_parts.append(segment)
        emitted = len(response.tokens)
        tokens += emitted
        if emitted:
            acceptance_lengths.append(response.accepted)
        last_tps = float(response.generation_tps)
        peak_memory = max(peak_memory, float(response.peak_memory))
    runtime.mx.synchronize()
    elapsed = time.perf_counter() - start

    if stream:
        print("\n", flush=True)

    return RunStats(
        name="dflash",
        text="".join(text_parts),
        tokens=tokens,
        elapsed_s=elapsed,
        generation_tps=last_tps,
        peak_memory_gb=peak_memory,
        acceptance_lengths=acceptance_lengths,
    )


def _warmup(
    runtime: SimpleNamespace,
    model: Any,
    draft: Any | None,
    tokenizer: Any,
    sampler: Any,
    block_size: int,
    warmup_tokens: int,
    *,
    include_target: bool,
    include_dflash: bool,
) -> None:
    if warmup_tokens <= 0:
        return
    warmup_prompt = tokenizer.encode("Hi")
    if include_target:
        list(runtime.stream_generate_baseline(model, tokenizer, warmup_prompt, warmup_tokens, sampler=sampler))
    if include_dflash:
        if draft is None:
            raise ValueError("DFlash warmup requires a loaded draft model")
        list(
            runtime.stream_generate(
                model,
                draft,
                tokenizer,
                warmup_prompt,
                block_size=block_size,
                max_tokens=warmup_tokens,
                sampler=sampler,
            )
        )
    runtime.mx.synchronize()


def _print_stats(target: RunStats | None, dflash: RunStats | None, *, block_size: int, exact_match: bool) -> None:
    print("Stats")
    print("-----")
    for stats in [target, dflash]:
        if stats is None:
            continue
        print(
            f"{stats.name:11} "
            f"tokens={stats.tokens:4d} "
            f"time={stats.elapsed_s:7.3f}s "
            f"tps={stats.measured_tps:7.2f} "
            f"peak={stats.peak_memory_gb:5.2f} GB"
        )

    if target is not None and dflash is not None:
        speedup = dflash.measured_tps / max(target.measured_tps, 1e-9)
        print(f"speedup     {speedup:7.2f}x")
        print(f"same text   {'yes' if exact_match else 'no'}")

    if dflash is not None and dflash.acceptance_lengths:
        accepted = dflash.acceptance_lengths
        avg_accept = sum(accepted) / len(accepted)
        full_blocks = sum(1 for value in accepted if value >= block_size)
        print(f"avg accept  {avg_accept:7.2f} tokens/block")
        print(f"full block  {full_blocks / len(accepted) * 100:7.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream one MLX prompt with target-only and DFlash timing.")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Target model (default: {DEFAULT_MODEL})")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL, help=f"DFlash draft model (default: {DEFAULT_DRAFT_MODEL})")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-prompt", action="store_true", help="Use --prompt directly without chat template")
    parser.add_argument("--dflash", action="store_true", help="Stream the DFlash run instead of the target-only run")
    parser.add_argument("--no-compare", action="store_true", help="Only run the streamed path; omit speedup comparison")
    parser.add_argument("--warmup-tokens", type=int, default=3, help="Warmup tokens per path before measuring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = _load_runtime()
    run_target = (not args.dflash) or (not args.no_compare)
    run_dflash = args.dflash or (not args.no_compare)

    print(f"Loading target: {args.model}", file=sys.stderr)
    model, tokenizer = runtime.load(args.model)
    draft = None
    if run_dflash:
        print(f"Loading draft:  {args.draft_model}", file=sys.stderr)
        draft = runtime.load_draft(args.draft_model)

    sampler = runtime.make_sampler(temp=args.temperature)
    prompt = _make_prompt(
        tokenizer,
        args.prompt,
        enable_thinking=args.enable_thinking,
        raw_prompt=args.raw_prompt,
    )

    _warmup(
        runtime,
        model,
        draft,
        tokenizer,
        sampler,
        args.block_size,
        args.warmup_tokens,
        include_target=run_target,
        include_dflash=run_dflash,
    )

    target_stats: RunStats | None = None
    dflash_stats: RunStats | None = None

    if args.dflash:
        if draft is None:
            raise ValueError("--dflash requires a loaded draft model")
        dflash_stats = _run_dflash(
            runtime,
            model,
            draft,
            tokenizer,
            prompt,
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            sampler=sampler,
            stream=True,
        )
        if not args.no_compare:
            print("Running target-only comparison...", file=sys.stderr)
            target_stats = _run_target(
                runtime,
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                stream=False,
            )
    else:
        target_stats = _run_target(
            runtime,
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            stream=True,
        )
        if not args.no_compare:
            print("Running DFlash comparison...", file=sys.stderr)
            if draft is None:
                raise ValueError("DFlash comparison requires a loaded draft model")
            dflash_stats = _run_dflash(
                runtime,
                model,
                draft,
                tokenizer,
                prompt,
                block_size=args.block_size,
                max_tokens=args.max_tokens,
                sampler=sampler,
                stream=False,
            )

    exact_match = bool(target_stats and dflash_stats and target_stats.text == dflash_stats.text)
    _print_stats(target_stats, dflash_stats, block_size=args.block_size, exact_match=exact_match)


if __name__ == "__main__":
    main()
