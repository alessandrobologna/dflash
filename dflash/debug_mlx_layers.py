from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from rich import print
from tqdm import tqdm

from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .benchmark import DEFAULT_SAMPLE_SEED, _apply_chat_template, _limit_dataset, load_and_process_dataset
from .model_mlx_clean import (
    _target_core,
    _target_lm_head,
    disable_gated_delta_kernel,
    load,
    load_draft,
)
from .verify_mlx_target import (
    _accepted_drafts,
    _argmax_token,
    _draft_block,
    _first_mismatch,
    _prefill_target,
    _replay_accepted,
    _sequential_posterior,
)


@dataclass
class LayerDiff:
    layer: int
    kind: str
    max_abs: float
    mean_abs: float
    block_l2: float
    sequential_l2: float


@dataclass
class StageDiff:
    stage: str
    max_abs: float
    mean_abs: float


@dataclass
class LayerTraceResult:
    sample: int
    turn: int
    step: int
    generated: int
    prompt_sha1: str
    prompt_preview: str
    first_posterior_mismatch: int
    block_acceptance: int
    sequential_acceptance: int
    block_pending: int
    sequential_pending: int
    block_tokens: list[int]
    block_posterior: list[int]
    sequential_posterior: list[int]
    first_nonzero_layer: int | None
    first_material_layer: int | None
    first_gdn_stage: str | None
    layer_diffs: list[LayerDiff]
    gdn_stage_diffs: list[StageDiff]


def _forward_trace(target: Any, inputs: mx.array, cache: list[Any]) -> tuple[list[mx.array], mx.array]:
    core = _target_core(target)
    hidden_states = core.embed_tokens(inputs)
    traces = []

    if hasattr(core, "fa_idx") and hasattr(core, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[core.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[core.ssm_idx])
        for layer, layer_cache in zip(core.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            traces.append(hidden_states)
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer, layer_cache in zip(core.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            traces.append(hidden_states)

    logits = _target_lm_head(target, core.norm(hidden_states))
    mx.eval(logits, *traces)
    return traces, logits


def _sequential_trace(target: Any, block_tokens: mx.array, cache: list[Any]) -> tuple[list[mx.array], mx.array]:
    num_layers = len(_target_core(target).layers)
    layer_chunks: list[list[mx.array]] = [[] for _ in range(num_layers)]
    logits_chunks = []
    for idx in range(block_tokens.shape[1]):
        traces, logits = _forward_trace(target, block_tokens[:, idx : idx + 1], cache)
        for layer_idx, trace in enumerate(traces):
            layer_chunks[layer_idx].append(trace)
        logits_chunks.append(logits)
    return [mx.concatenate(chunks, axis=1) for chunks in layer_chunks], mx.concatenate(logits_chunks, axis=1)


def _layer_diffs(
    target: Any,
    checkpoint: list[Any],
    block_tokens: mx.array,
    position: int,
) -> list[LayerDiff]:
    block_cache = copy.deepcopy(checkpoint)
    sequential_cache = copy.deepcopy(checkpoint)
    block_traces, _ = _forward_trace(target, block_tokens, block_cache)
    sequential_traces, _ = _sequential_trace(target, block_tokens, sequential_cache)

    diffs = []
    for idx, (block_trace, sequential_trace) in enumerate(zip(block_traces, sequential_traces)):
        delta = block_trace[:, position : position + 1, :].astype(mx.float32) - sequential_trace[
            :, position : position + 1, :
        ].astype(mx.float32)
        block_pos = block_trace[:, position : position + 1, :].astype(mx.float32)
        sequential_pos = sequential_trace[:, position : position + 1, :].astype(mx.float32)
        mx.eval(delta, block_pos, sequential_pos)
        layer = _target_core(target).layers[idx]
        diffs.append(
            LayerDiff(
                layer=idx,
                kind="linear" if layer.is_linear else "attention",
                max_abs=float(mx.max(mx.abs(delta)).item()),
                mean_abs=float(mx.mean(mx.abs(delta)).item()),
                block_l2=float(mx.sqrt(mx.sum(block_pos * block_pos)).item()),
                sequential_l2=float(mx.sqrt(mx.sum(sequential_pos * sequential_pos)).item()),
            )
        )
    return diffs


def _gdn_trace(layer: Any, x: mx.array, mask: mx.array | None, cache: Any) -> tuple[mx.array, dict[str, mx.array]]:
    import mlx_lm.models.gated_delta as gated_delta
    import mlx_lm.models.qwen3_5 as qwen3_5

    attn = layer.linear_attn
    inputs = layer.input_layernorm(x)
    batch, seq_len, _ = inputs.shape
    qkv = attn.in_proj_qkv(inputs)
    z = attn.in_proj_z(inputs).reshape(batch, seq_len, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(inputs)
    a = attn.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((batch, attn.conv_kernel_size - 1, attn.conv_dim), dtype=inputs.dtype)

    masked_qkv = mx.where(mask[..., None], qkv, 0) if mask is not None else qkv
    conv_input = mx.concatenate([conv_state, masked_qkv], axis=1)
    if cache is not None:
        n_keep = attn.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, seq_len)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(attn.conv1d(conv_input))

    q, k, v = [
        value.reshape(batch, seq_len, heads, dim)
        for value, heads, dim in zip(
            mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
            [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
            [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q_normed = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k_normed = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    recurrent_out, state = qwen3_5.gated_delta_update(
        q_normed,
        k_normed,
        v,
        a,
        b,
        attn.A_log,
        attn.dt_bias,
        state,
        mask,
        use_kernel=not attn.training,
    )
    if cache is not None:
        cache[1] = state
        cache.advance(seq_len)

    gated_norm = attn.norm(recurrent_out, z)
    projected = attn.out_proj(gated_norm.reshape(batch, seq_len, -1))
    layer_output = x + projected
    layer_output = layer_output + layer.mlp(layer.post_attention_layernorm(layer_output))
    traces = {
        "input_norm": inputs,
        "qkv": qkv,
        "masked_qkv": masked_qkv,
        "conv_out": conv_out,
        "q": q_normed,
        "k": k_normed,
        "v": v,
        "a": a,
        "b": b,
        "g": gated_delta.compute_g(attn.A_log, a, attn.dt_bias),
        "recurrent_out": recurrent_out,
        "gated_norm": gated_norm,
        "projected": projected,
        "layer_output": layer_output,
    }
    mx.eval(layer_output, *traces.values())
    return layer_output, traces


def _first_gdn_stage_diffs(
    target: Any,
    checkpoint: list[Any],
    block_tokens: mx.array,
    position: int,
) -> list[StageDiff]:
    core = _target_core(target)
    layer_idx = next(idx for idx, layer in enumerate(core.layers) if layer.is_linear)
    layer = core.layers[layer_idx]

    block_cache = copy.deepcopy(checkpoint)
    block_hidden = core.embed_tokens(block_tokens)
    block_mask = create_ssm_mask(block_hidden, block_cache[core.ssm_idx])
    _, block_traces = _gdn_trace(layer, block_hidden, block_mask, block_cache[layer_idx])

    sequential_cache = copy.deepcopy(checkpoint)
    seq_chunks: dict[str, list[mx.array]] = {}
    for idx in range(block_tokens.shape[1]):
        seq_hidden = core.embed_tokens(block_tokens[:, idx : idx + 1])
        seq_mask = create_ssm_mask(seq_hidden, sequential_cache[core.ssm_idx])
        _, step_traces = _gdn_trace(layer, seq_hidden, seq_mask, sequential_cache[layer_idx])
        for name, value in step_traces.items():
            seq_chunks.setdefault(name, []).append(value)

    diffs = []
    for name, block_value in block_traces.items():
        block_pos = block_value[:, position : position + 1]
        sequential_value = mx.concatenate(seq_chunks[name], axis=1)
        sequential_pos = sequential_value[:, position : position + 1]
        delta = block_pos.astype(mx.float32) - sequential_pos.astype(mx.float32)
        mx.eval(delta)
        diffs.append(
            StageDiff(
                stage=name,
                max_abs=float(mx.max(mx.abs(delta)).item()),
                mean_abs=float(mx.mean(mx.abs(delta)).item()),
            )
        )
    return diffs


def _block_posterior_from_checkpoint(
    target: Any,
    block_tokens: mx.array,
    checkpoint: list[Any],
) -> list[int]:
    traces, logits = _forward_trace(target, block_tokens, copy.deepcopy(checkpoint))
    del traces
    posterior = _argmax_token(logits).tolist()[0]
    mx.eval(logits)
    return posterior


def _find_trace_for_prompt(
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
    material_tolerance: float,
) -> LayerTraceResult | None:
    prompt_tokens = mx.array(tokenizer.encode(prompt), dtype=mx.uint32)
    target_cache, target_hidden, pending = _prefill_target(target, prompt_tokens, draft.target_layer_ids)
    draft_cache = draft.make_cache()
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    generated = 0
    step = 0
    pending_emitted = False
    while generated < max_tokens:
        block_tokens = _draft_block(target, draft, draft_cache, target_hidden, pending, block_size)
        block_token_list = block_tokens.tolist()[0]
        checkpoint = copy.deepcopy(target_cache)
        block_post = _block_posterior_from_checkpoint(target, block_tokens, checkpoint)
        seq_cache = copy.deepcopy(checkpoint)
        seq_post, _ = _sequential_posterior(target, block_tokens, seq_cache, draft.target_layer_ids)

        block_accept = _accepted_drafts(block_token_list, block_post)
        seq_accept = _accepted_drafts(block_token_list, seq_post)
        block_pending = block_post[block_accept]
        seq_pending = seq_post[seq_accept]
        first_mismatch = _first_mismatch(block_post, seq_post)
        if first_mismatch is not None and (block_accept != seq_accept or block_pending != seq_pending):
            diffs = _layer_diffs(target, checkpoint, block_tokens, first_mismatch)
            gdn_diffs = _first_gdn_stage_diffs(target, checkpoint, block_tokens, first_mismatch)
            first_nonzero = next((diff.layer for diff in diffs if diff.max_abs > 0.0), None)
            first_material = next((diff.layer for diff in diffs if diff.max_abs >= material_tolerance), None)
            first_gdn_stage = next((diff.stage for diff in gdn_diffs if diff.max_abs > 0.0), None)
            return LayerTraceResult(
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
                first_nonzero_layer=first_nonzero,
                first_material_layer=first_material,
                first_gdn_stage=first_gdn_stage,
                layer_diffs=diffs,
                gdn_stage_diffs=gdn_diffs,
            )

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

    return None


def run(args: argparse.Namespace) -> None:
    if args.disable_gated_delta_kernel:
        disable_gated_delta_kernel()
        print("GatedDeltaNet kernel disabled: using use_kernel=False")

    target, tokenizer = load(args.model)
    draft = load_draft(args.draft_model)
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)
    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples, args.sample_seed)

    for sample_idx, instance in enumerate(tqdm(dataset)):
        messages = []
        for turn_idx, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            result = _find_trace_for_prompt(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_preview=user_content,
                sample_idx=sample_idx,
                turn_idx=turn_idx,
                block_size=block_size,
                max_tokens=args.max_new_tokens,
                material_tolerance=args.material_tolerance,
            )
            if result is not None:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
                return

    print("No block/sequential decision mismatch found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace first MLX target layer divergence for a DFlash block")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-gated-delta-kernel", action="store_true")
    parser.add_argument("--material-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
