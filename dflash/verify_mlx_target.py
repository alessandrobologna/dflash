from __future__ import annotations

import argparse
import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from rich import print
from tqdm import tqdm

from .benchmark import (
    DEFAULT_SAMPLE_SEED,
    _apply_chat_template,
    _limit_dataset,
    load_and_process_dataset,
)
from .model_mlx_clean import (
    _target_embedding,
    _target_forward,
    _target_lm_head,
    disable_gated_delta_kernel,
    load,
    load_draft,
)


@dataclass
class VerifyFailure:
    sample: int
    turn: int
    step: int
    generated: int
    prompt_sha1: str
    prompt_preview: str
    first_posterior_mismatch: int | None
    block_acceptance: int
    sequential_acceptance: int
    block_pending: int
    sequential_pending: int
    block_tokens: list[int]
    block_posterior: list[int]
    sequential_posterior: list[int]


def _argmax_token(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.uint32)


def _accepted_drafts(block_tokens: list[int], posterior: list[int]) -> int:
    accepted = 0
    for idx, proposed in enumerate(block_tokens[1:]):
        if proposed != posterior[idx]:
            break
        accepted += 1
    return accepted


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _prefill_target(target: Any, prompt: mx.array, layer_ids: list[int]):
    from mlx_lm.models import cache as cache_utils

    target_cache = cache_utils.make_prompt_cache(target)
    target_hidden = None
    while prompt.size > 1:
        logits, target_hidden = _target_forward(target, prompt[:-1][None], target_cache, layer_ids)
        mx.eval(logits, target_hidden)
        prompt = prompt[-1:]

    logits, target_hidden = _target_forward(target, prompt[None], target_cache, layer_ids)
    pending = int(_argmax_token(logits[:, -1, :]).item())
    mx.eval(target_hidden)
    return target_cache, target_hidden, pending


def _draft_block(target: Any, draft: Any, draft_cache: list[Any], target_hidden: mx.array, pending: int, block_size: int):
    from mlx_lm.models import cache as cache_utils

    block_tokens = mx.array([[pending] + [draft.mask_token_id] * (block_size - 1)], dtype=mx.uint32)
    if block_size > 1:
        noise_embedding = _target_embedding(target)(block_tokens)
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            cache=draft_cache,
        )
        draft_logits = _target_lm_head(target, draft_hidden[:, 1:, :])
        sampled_draft = _argmax_token(draft_logits)
        block_tokens = mx.concatenate([block_tokens[:, :1], sampled_draft], axis=1)
        cache_utils.trim_prompt_cache(draft_cache, block_size)
    mx.eval(block_tokens)
    return block_tokens


def _block_posterior(target: Any, block_tokens: mx.array, target_cache: list[Any], layer_ids: list[int]) -> list[int]:
    logits, hidden = _target_forward(target, block_tokens, target_cache, layer_ids)
    posterior = _argmax_token(logits).tolist()[0]
    mx.eval(hidden)
    return posterior


def _sequential_posterior(
    target: Any,
    block_tokens: mx.array,
    target_cache: list[Any],
    layer_ids: list[int],
) -> tuple[list[int], mx.array]:
    posterior: list[int] = []
    hiddens = []
    for idx in range(block_tokens.shape[1]):
        logits, hidden = _target_forward(
            target,
            block_tokens[:, idx : idx + 1],
            target_cache,
            layer_ids,
        )
        token = int(_argmax_token(logits[:, -1, :]).item())
        mx.eval(hidden)
        posterior.append(token)
        hiddens.append(hidden)
    return posterior, mx.concatenate(hiddens, axis=1)


def _replay_accepted(
    target: Any,
    checkpoint: list[Any],
    block_tokens: mx.array,
    accepted_count: int,
    layer_ids: list[int],
):
    target_cache = copy.deepcopy(checkpoint)
    hiddens = []
    for idx in range(accepted_count):
        _, hidden = _target_forward(
            target,
            block_tokens[:, idx : idx + 1],
            target_cache,
            layer_ids,
        )
        mx.eval(hidden)
        hiddens.append(hidden)
    return target_cache, mx.concatenate(hiddens, axis=1)


def _run_prompt_check(
    *,
    target: Any,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    prompt_preview: str,
    sample_idx: int,
    turn_idx: int,
    block_size: int,
    max_tokens: int,
    max_failures: int,
) -> tuple[int, int, float, float, list[VerifyFailure]]:
    prompt_tokens = mx.array(tokenizer.encode(prompt), dtype=mx.uint32)
    target_cache, target_hidden, pending = _prefill_target(target, prompt_tokens, draft.target_layer_ids)
    draft_cache = draft.make_cache()

    generated = 0
    step = 0
    pending_emitted = False
    checked_positions = 0
    posterior_mismatches = 0
    block_verify_s = 0.0
    sequential_verify_s = 0.0
    failures: list[VerifyFailure] = []
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    while generated < max_tokens:
        block_tokens = _draft_block(target, draft, draft_cache, target_hidden, pending, block_size)
        block_token_list = block_tokens.tolist()[0]

        checkpoint = copy.deepcopy(target_cache)
        mx.synchronize()
        block_start = time.perf_counter()
        block_cache = copy.deepcopy(checkpoint)
        block_post = _block_posterior(target, block_tokens, block_cache, draft.target_layer_ids)
        mx.synchronize()
        block_verify_s += time.perf_counter() - block_start

        mx.synchronize()
        sequential_start = time.perf_counter()
        seq_cache = copy.deepcopy(checkpoint)
        seq_post, _ = _sequential_posterior(target, block_tokens, seq_cache, draft.target_layer_ids)
        mx.synchronize()
        sequential_verify_s += time.perf_counter() - sequential_start

        checked_positions += len(block_post)
        first_mismatch = _first_mismatch(block_post, seq_post)
        if first_mismatch is not None:
            posterior_mismatches += 1

        block_accept = _accepted_drafts(block_token_list, block_post)
        seq_accept = _accepted_drafts(block_token_list, seq_post)
        block_pending = block_post[block_accept]
        seq_pending = seq_post[seq_accept]

        if block_accept != seq_accept or block_pending != seq_pending:
            failures.append(
                VerifyFailure(
                    sample=sample_idx,
                    turn=turn_idx,
                    step=step,
                    generated=generated,
                    prompt_sha1=prompt_hash,
                    prompt_preview=prompt_preview[:160].replace("\n", "\\n"),
                    first_posterior_mismatch=first_mismatch,
                    block_acceptance=block_accept + 1,
                    sequential_acceptance=seq_accept + 1,
                    block_pending=block_pending,
                    sequential_pending=seq_pending,
                    block_tokens=block_token_list,
                    block_posterior=block_post,
                    sequential_posterior=seq_post,
                )
            )
            if len(failures) >= max_failures:
                break

        accepted_count = seq_accept + 1
        pending = seq_pending
        target_cache, target_hidden = _replay_accepted(
            target,
            checkpoint,
            block_tokens,
            accepted_count,
            draft.target_layer_ids,
        )

        emitted = block_token_list[:accepted_count] if not pending_emitted else block_token_list[1:accepted_count]
        emitted.append(pending)
        pending_emitted = True
        generated += min(len(emitted), max_tokens - generated)
        step += 1

    return checked_positions, posterior_mismatches, block_verify_s, sequential_verify_s, failures


def run(args: argparse.Namespace) -> None:
    if args.disable_gated_delta_kernel:
        disable_gated_delta_kernel()
        print("GatedDeltaNet kernel disabled: using use_kernel=False")

    logger_args = (args.model, args.draft_model)
    print(f"Loading target: {logger_args[0]}")
    target, tokenizer = load(args.model)
    print(f"Loading draft:  {logger_args[1]}")
    draft = load_draft(args.draft_model)
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples, args.sample_seed)
    total_positions = 0
    total_posterior_mismatches = 0
    total_block_verify_s = 0.0
    total_sequential_verify_s = 0.0
    turns_checked = 0
    failures: list[VerifyFailure] = []

    for sample_idx, instance in enumerate(tqdm(dataset)):
        messages = []
        for turn_idx, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            checked, mismatches, block_s, sequential_s, prompt_failures = _run_prompt_check(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_preview=user_content,
                sample_idx=sample_idx,
                turn_idx=turn_idx,
                block_size=block_size,
                max_tokens=args.max_new_tokens,
                max_failures=max(args.max_failures - len(failures), 1),
            )
            total_positions += checked
            total_posterior_mismatches += mismatches
            total_block_verify_s += block_s
            total_sequential_verify_s += sequential_s
            turns_checked += 1
            failures.extend(prompt_failures)
            if len(failures) >= args.max_failures:
                break
        if len(failures) >= args.max_failures:
            break

    print("\nTarget verifier check:")
    print(f"  turns checked: {turns_checked}")
    print(f"  block size: {block_size}")
    print(f"  checked positions: {total_positions}")
    print(f"  posterior-mismatching blocks: {total_posterior_mismatches}")
    print(f"  block verify time: {total_block_verify_s:.3f}s")
    print(f"  sequential reference time: {total_sequential_verify_s:.3f}s")

    if failures:
        print(f"  decision mismatches: {len(failures)}")
        for failure in failures:
            print(f"  {failure}")
        raise AssertionError("Batched target verification changed a speculative accept/pending decision")

    print("  decision mismatches: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MLX target block verifier exactness")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--disable-gated-delta-kernel", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
