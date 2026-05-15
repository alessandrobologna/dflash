from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any

import mlx.core as mx
import numpy as np
from rich import print
from tqdm import tqdm

from .benchmark import DEFAULT_SAMPLE_SEED, _apply_chat_template, _limit_dataset, load_and_process_dataset
from .model_mlx_clean import _target_forward, disable_gated_delta_kernel, load, load_draft


@dataclass
class Divergence:
    block_size: int
    sample: int
    turn: int
    prefix_tokens: int
    block_position: int
    prompt_sha1: str
    prompt_preview: str
    mode: str
    block_tokens: list[int]
    block_token_origin: list[int]
    block_argmax: int
    sequential_argmax: int
    block_top5: list[tuple[int, float]]
    sequential_top5: list[tuple[int, float]]
    block_margin: float
    sequential_margin: float
    max_abs_logit_diff: float
    prefix_cache_summary: dict[str, Any]
    block_cache_summary: dict[str, Any]
    sequential_cache_summary: dict[str, Any]
    cache_details: dict[str, Any] | None


def _argmax(logits: mx.array) -> int:
    return int(mx.argmax(logits, axis=-1).item())


def _top5(logits: mx.array) -> tuple[list[tuple[int, float]], float]:
    values = logits.astype(mx.float32).reshape(-1)
    indices = mx.argsort(values)[-5:].tolist()[::-1]
    top = [(int(idx), float(values[int(idx)].item())) for idx in indices]
    margin = float(top[0][1] - top[1][1]) if len(top) > 1 else float("inf")
    return top, margin


def _cache_invariants(cache: list[Any]) -> list[dict[str, Any]]:
    invariants = []
    for idx, entry in enumerate(cache):
        item: dict[str, Any] = {
            "idx": idx,
            "class": entry.__class__.__name__,
        }
        if hasattr(entry, "size"):
            try:
                item["size"] = int(entry.size())
            except TypeError:
                item["size"] = str(entry.size())
        if hasattr(entry, "offset"):
            try:
                item["offset"] = int(entry.offset)
            except TypeError:
                item["offset"] = str(entry.offset)
        if hasattr(entry, "lengths"):
            lengths = entry.lengths
            item["lengths"] = None if lengths is None else np.asarray(lengths).tolist()
        if hasattr(entry, "left_padding"):
            left_padding = entry.left_padding
            item["left_padding"] = None if left_padding is None else np.asarray(left_padding).tolist()
        if hasattr(entry, "cache"):
            item["slots"] = [
                None
                if value is None
                else {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for value in entry.cache
            ]
        else:
            state = getattr(entry, "state", None)
            if isinstance(state, tuple):
                item["state"] = [
                    None
                    if value is None
                    else {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for value in state
                ]
        invariants.append(item)
    return invariants


def _cache_summary(cache: list[Any]) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    slot_shape_counts: dict[str, int] = {}
    state_shape_counts: dict[str, int] = {}
    sizes: list[int] = []
    offsets: list[int] = []

    for idx, entry in enumerate(cache):
        cls = entry.__class__.__name__
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if hasattr(entry, "size"):
            try:
                size = int(entry.size())
                sizes.append(size)
            except TypeError:
                pass
        if hasattr(entry, "offset"):
            try:
                offset = int(entry.offset)
                offsets.append(offset)
            except TypeError:
                pass
        if hasattr(entry, "cache"):
            for slot_idx, value in enumerate(entry.cache):
                if value is None:
                    continue
                signature = f"slot{slot_idx}:{tuple(value.shape)}:{value.dtype}"
                slot_shape_counts[signature] = slot_shape_counts.get(signature, 0) + 1
        else:
            state = getattr(entry, "state", None)
            if isinstance(state, tuple):
                for state_idx, value in enumerate(state):
                    if value is None:
                        continue
                    signature = f"slot{state_idx}:{tuple(value.shape)}:{value.dtype}"
                    state_shape_counts[signature] = state_shape_counts.get(signature, 0) + 1

    return {
        "entries": len(cache),
        "classes": class_counts,
        "sizes": sorted(set(sizes)),
        "offsets": sorted(set(offsets)),
        "slot_shapes": slot_shape_counts,
        "state_shapes": state_shape_counts,
    }


def _prefill_prefix(target: Any, tokens: list[int], layer_ids: list[int]):
    from mlx_lm.models import cache as cache_utils

    cache = cache_utils.make_prompt_cache(target)
    hidden = None
    prompt = mx.array(tokens, dtype=mx.uint32)
    while prompt.size > 1:
        logits, hidden = _target_forward(target, prompt[:-1][None], cache, layer_ids)
        mx.eval(logits, hidden)
        prompt = prompt[-1:]
    logits, hidden = _target_forward(target, prompt[None], cache, layer_ids)
    next_token = _argmax(logits[:, -1, :])
    mx.eval(hidden)
    return cache, hidden, next_token


def _target_sequence(target: Any, prompt_tokens: list[int], layer_ids: list[int], max_tokens: int) -> list[int]:
    cache, _, next_token = _prefill_prefix(target, prompt_tokens, layer_ids)
    generated = []
    token = next_token
    for _ in range(max_tokens):
        generated.append(token)
        logits, hidden = _target_forward(target, mx.array([[token]], dtype=mx.uint32), cache, layer_ids)
        token = _argmax(logits[:, -1, :])
        mx.eval(hidden)
    return generated


def _build_block(true_tokens: list[int], start: int, block_size: int, mode: str) -> tuple[list[int], list[int]]:
    block = list(true_tokens[start : start + block_size])
    origin = list(block)
    if mode == "true":
        return block, origin
    if mode == "perturb-tail" and len(block) > 2:
        perturb_at = min(len(block) - 1, 2)
        block[perturb_at] = (block[perturb_at] + 7919) % 151936
    elif mode == "perturb-after-4" and len(block) > 5:
        perturb_at = min(len(block) - 1, 5)
        block[perturb_at] = (block[perturb_at] + 7919) % 151936
    return block, origin


def _block_logits(target: Any, cache: list[Any], block: list[int], layer_ids: list[int]) -> mx.array:
    logits, hidden = _target_forward(target, mx.array([block], dtype=mx.uint32), cache, layer_ids)
    mx.eval(logits, hidden)
    return logits[0]


def _sequential_logits(target: Any, cache: list[Any], block: list[int], layer_ids: list[int]) -> mx.array:
    outputs = []
    for token in block:
        logits, hidden = _target_forward(target, mx.array([[token]], dtype=mx.uint32), cache, layer_ids)
        mx.eval(logits, hidden)
        outputs.append(logits[:, -1, :])
    return mx.concatenate(outputs, axis=0)


def _find_divergence(
    *,
    target: Any,
    prompt_tokens: list[int],
    prompt_preview: str,
    sample_idx: int,
    turn_idx: int,
    true_tokens: list[int],
    layer_ids: list[int],
    block_size: int,
    mode: str,
    max_prefixes: int,
    dump_cache_invariants: bool,
) -> Divergence | None:
    prompt_hash = hashlib.sha1(json.dumps(prompt_tokens).encode("utf-8")).hexdigest()[:12]
    max_start = max(0, len(true_tokens) - block_size)
    starts = list(range(0, max_start + 1, block_size))
    if max_prefixes > 0:
        starts = starts[:max_prefixes]

    for start in starts:
        prefix = prompt_tokens + true_tokens[:start]
        block, origin = _build_block(true_tokens, start, block_size, mode)
        if len(block) != block_size:
            continue

        prefix_cache, _, _ = _prefill_prefix(target, prefix, layer_ids)
        block_cache = copy.deepcopy(prefix_cache)
        sequential_cache = copy.deepcopy(prefix_cache)
        block_logits = _block_logits(target, block_cache, block, layer_ids)
        sequential_logits = _sequential_logits(target, sequential_cache, block, layer_ids)

        block_argmax = mx.argmax(block_logits, axis=-1).tolist()
        sequential_argmax = mx.argmax(sequential_logits, axis=-1).tolist()
        for pos, (block_token, sequential_token) in enumerate(zip(block_argmax, sequential_argmax)):
            if int(block_token) == int(sequential_token):
                continue
            block_vec = block_logits[pos]
            sequential_vec = sequential_logits[pos]
            block_top5, block_margin = _top5(block_vec)
            sequential_top5, sequential_margin = _top5(sequential_vec)
            diff = block_vec.astype(mx.float32) - sequential_vec.astype(mx.float32)
            max_abs_diff = float(mx.max(mx.abs(diff)).item())
            cache_details = None
            if dump_cache_invariants:
                cache_details = {
                    "prefix": _cache_invariants(prefix_cache),
                    "block_after": _cache_invariants(block_cache),
                    "sequential_after": _cache_invariants(sequential_cache),
                }
            return Divergence(
                block_size=block_size,
                sample=sample_idx,
                turn=turn_idx,
                prefix_tokens=start,
                block_position=pos,
                prompt_sha1=prompt_hash,
                prompt_preview=prompt_preview[:160].replace("\n", "\\n"),
                mode=mode,
                block_tokens=block,
                block_token_origin=origin,
                block_argmax=int(block_token),
                sequential_argmax=int(sequential_token),
                block_top5=block_top5,
                sequential_top5=sequential_top5,
                block_margin=block_margin,
                sequential_margin=sequential_margin,
                max_abs_logit_diff=max_abs_diff,
                prefix_cache_summary=_cache_summary(prefix_cache),
                block_cache_summary=_cache_summary(block_cache),
                sequential_cache_summary=_cache_summary(sequential_cache),
                cache_details=cache_details,
            )
    return None


def run(args: argparse.Namespace) -> None:
    if args.disable_gated_delta_kernel:
        disable_gated_delta_kernel()
        print("GatedDeltaNet kernel disabled: using use_kernel=False")

    target, tokenizer = load(args.model)
    draft = load_draft(args.draft_model)
    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples, args.sample_seed)
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    divergences: list[Divergence] = []
    for sample_idx, instance in enumerate(tqdm(dataset)):
        messages = []
        for turn_idx, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            prompt_tokens = tokenizer.encode(prompt)
            true_tokens = _target_sequence(target, prompt_tokens, draft.target_layer_ids, args.max_new_tokens + max(block_sizes))
            for block_size in block_sizes:
                for mode in modes:
                    divergence = _find_divergence(
                        target=target,
                        prompt_tokens=prompt_tokens,
                        prompt_preview=user_content,
                        sample_idx=sample_idx,
                        turn_idx=turn_idx,
                        true_tokens=true_tokens,
                        layer_ids=draft.target_layer_ids,
                        block_size=block_size,
                        mode=mode,
                        max_prefixes=args.max_prefixes,
                        dump_cache_invariants=args.dump_cache_invariants,
                    )
                    if divergence is not None:
                        divergences.append(divergence)
                        if len(divergences) >= args.max_failures:
                            _print_summary(divergences)
                            raise AssertionError("Target-only block-vs-sequential divergence found")
            if len(divergences) >= args.max_failures:
                break
        if len(divergences) >= args.max_failures:
            break

    _print_summary(divergences)
    if divergences:
        raise AssertionError("Target-only block-vs-sequential divergence found")


def _print_summary(divergences: list[Divergence]) -> None:
    print("\nTarget-only block-vs-sequential check:")
    print(f"  divergences: {len(divergences)}")
    for divergence in divergences:
        print(json.dumps(asdict(divergence), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug MLX target block-vs-sequential decode equivalence")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--modes", type=str, default="true,perturb-tail,perturb-after-4")
    parser.add_argument("--max-prefixes", type=int, default=0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-gated-delta-kernel", action="store_true")
    parser.add_argument("--dump-cache-invariants", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
