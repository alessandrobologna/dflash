from __future__ import annotations

import copy
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load as mlx_lm_load
from mlx_lm.models import cache as cache_utils
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.tokenizer_utils import TokenizerWrapper


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _as_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(
        snapshot_download(
            path_or_repo,
            allow_patterns=["*.json", "model*.safetensors"],
        )
    )


def _load_config(model_path: Path) -> dict[str, Any]:
    with open(model_path / "config.json") as f:
        return json.load(f)


def _sample(logits: mx.array, sampler: Optional[Callable[[mx.array], mx.array]]) -> mx.array:
    if sampler is None:
        return mx.argmax(logits, axis=-1)

    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_logprobs = flat_logits - mx.logsumexp(flat_logits, axis=-1, keepdims=True)
    return sampler(flat_logprobs).reshape(logits.shape[:-1])


def _mark(enabled: bool) -> float:
    if not enabled:
        return 0.0
    mx.synchronize()
    return time.perf_counter()


def _elapsed(start: float, enabled: bool) -> float:
    if not enabled:
        return 0.0
    mx.synchronize()
    return time.perf_counter() - start


def disable_gated_delta_kernel() -> None:
    import mlx_lm.models.qwen3_5 as qwen3_5

    current = qwen3_5.gated_delta_update
    if getattr(current, "_dflash_reference_path", False):
        return

    def reference_gated_delta_update(*args, **kwargs):
        kwargs["use_kernel"] = False
        return current(*args, **kwargs)

    reference_gated_delta_update._dflash_reference_path = True
    qwen3_5.gated_delta_update = reference_gated_delta_update


@dataclass
class _GatedDeltaCommit:
    layer: Any
    cache: Any
    conv_input: mx.array
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array
    mask: Optional[mx.array]
    old_recurrent_state: Optional[mx.array]
    old_lengths: Optional[mx.array]
    old_left_padding: Optional[mx.array]
    seq_len: int


@dataclass
class DFlashModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    block_size: int
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: Optional[dict[str, Any]] = None
    sliding_window: Optional[int] = None
    layer_types: Optional[list[str]] = None
    dflash_config: Optional[dict[str, Any]] = None
    num_target_layers: Optional[int] = None


class DFlashAttention(nn.Module):
    def __init__(self, args: DFlashModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, q_out, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_out, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, kv_out, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_out, args.hidden_size, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        queries = self.q_proj(hidden_states)
        queries = queries.reshape(batch, q_len, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)

        keys = mx.concatenate([self.k_proj(target_hidden), self.k_proj(hidden_states)], axis=1)
        values = mx.concatenate([self.v_proj(target_hidden), self.v_proj(hidden_states)], axis=1)
        kv_len = keys.shape[1]
        keys = keys.reshape(batch, kv_len, self.n_kv_heads, self.head_dim)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.reshape(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        keys = self.rope(keys, offset=offset)
        queries = self.rope(queries, offset=offset + ctx_len)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=None,
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, q_len, -1)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    def __init__(self, args: DFlashModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DFlashModelArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = DFlashMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, args: DFlashModelArgs):
        super().__init__()
        self.args = args
        cfg = args.dflash_config or {}
        num_target_layers = args.num_target_layers or args.num_hidden_layers
        self.target_layer_ids = cfg.get(
            "target_layer_ids",
            build_target_layer_ids(num_target_layers, args.num_hidden_layers),
        )
        self.block_size = args.block_size
        self.mask_token_id = cfg.get("mask_token_id")
        self.layers = [DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(len(self.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            target_layer_ids=self.target_layer_ids,
        )

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        *,
        target_hidden: mx.array,
        noise_embedding: mx.array,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, target_hidden, layer_cache)
        return self.norm(hidden_states)


def load(path_or_hf_repo: str, **kwargs):
    return mlx_lm_load(path_or_hf_repo, **kwargs)


def load_draft(path_or_hf_repo: str, *, lazy: bool = False, strict: bool = True) -> DFlashDraftModel:
    model_path = _as_path(path_or_hf_repo)
    config = _load_config(model_path)
    model = DFlashDraftModel(DFlashModelArgs.from_dict(config))

    weights = {}
    for weight_file in glob.glob(str(model_path / "model*.safetensors")):
        weights.update(mx.load(weight_file))
    if not weights and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    return model


def _target_core(target: nn.Module) -> nn.Module:
    if hasattr(target, "language_model"):
        return target.language_model.model
    return target.model


def _target_layers(target: nn.Module) -> list[nn.Module]:
    return _target_core(target).layers


def _target_embedding(target: nn.Module):
    return _target_core(target).embed_tokens


def _target_lm_head(target: nn.Module, hidden_states: mx.array) -> mx.array:
    if hasattr(target, "language_model"):
        language_model = target.language_model
        if language_model.args.tie_word_embeddings:
            return language_model.model.embed_tokens.as_linear(hidden_states)
        return language_model.lm_head(hidden_states)

    if target.args.tie_word_embeddings:
        return target.model.embed_tokens.as_linear(hidden_states)
    return target.lm_head(hidden_states)


def _qwen35_gated_delta_with_commit(
    layer: Any,
    inputs: mx.array,
    mask: Optional[mx.array],
    cache: Optional[Any],
    commits: list[_GatedDeltaCommit],
) -> mx.array:
    from mlx_lm.models import qwen3_5

    if getattr(layer, "sharding_group", None) is not None:
        return layer(inputs, mask, cache)

    batch, seq_len, _ = inputs.shape
    qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(batch, seq_len, layer.num_v_heads, layer.head_v_dim)
    b = layer.in_proj_b(inputs)
    a = layer.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (batch, layer.conv_kernel_size - 1, layer.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)

    old_recurrent_state = cache[1] if cache is not None else None
    old_lengths = cache.lengths if cache is not None else None
    old_left_padding = cache.left_padding if cache is not None else None

    if cache is not None:
        n_keep = layer.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, seq_len)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    conv_out = nn.silu(layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(batch, seq_len, heads, dim)
        for t, heads, dim in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = qwen3_5.gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state,
        mask,
        use_kernel=not layer.training,
    )

    if cache is not None:
        cache[1] = state
        cache.advance(seq_len)
        commits.append(
            _GatedDeltaCommit(
                layer=layer,
                cache=cache,
                conv_input=conv_input,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                mask=mask,
                old_recurrent_state=old_recurrent_state,
                old_lengths=old_lengths,
                old_left_padding=old_left_padding,
                seq_len=seq_len,
            )
        )

    out = layer.norm(out, z)
    return layer.out_proj(out.reshape(batch, seq_len, -1))


def _qwen35_decoder_layer_with_commit(
    layer: Any,
    hidden_states: mx.array,
    mask: Optional[mx.array],
    cache: Optional[Any],
    commits: list[_GatedDeltaCommit],
) -> mx.array:
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    if layer.is_linear:
        attn_out = _qwen35_gated_delta_with_commit(layer.linear_attn, normed, mask, cache, commits)
    else:
        attn_out = layer.self_attn(normed, mask, cache)
    hidden_states = residual + attn_out
    return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))


def _can_commit_gated_delta_history(target: nn.Module, cache: list[Any]) -> bool:
    core = _target_core(target)
    if not (hasattr(core, "fa_idx") and hasattr(core, "ssm_idx")):
        return False
    return any(entry.__class__.__name__ == "ArraysCache" for entry in cache)


def _trim_trimmable_cache_entries_to(cache: list[Any], target_size: int) -> None:
    for entry in cache:
        if not hasattr(entry, "is_trimmable") or not entry.is_trimmable():
            continue
        current = entry.size()
        if current > target_size:
            entry.trim(current - target_size)


def _cache_arrays(cache: list[Any]) -> list[mx.array]:
    arrays = []
    for entry in cache:
        state = getattr(entry, "state", ())
        if state is None:
            continue
        if not isinstance(state, (list, tuple)):
            state = (state,)
        arrays.extend(item for item in state if item is not None)
    return arrays


def _apply_gated_delta_history_commits(
    commits: list[_GatedDeltaCommit],
    accepted_count: int,
) -> None:
    from mlx_lm.models import qwen3_5

    for commit in commits:
        if accepted_count >= commit.seq_len:
            continue

        n_keep = commit.layer.conv_kernel_size - 1
        commit.cache[0] = mx.contiguous(commit.conv_input[:, accepted_count : accepted_count + n_keep, :])
        prefix_mask = None if commit.mask is None else commit.mask[:, :accepted_count]
        _, state = qwen3_5.gated_delta_update(
            commit.q[:, :accepted_count],
            commit.k[:, :accepted_count],
            commit.v[:, :accepted_count],
            commit.a[:, :accepted_count],
            commit.b[:, :accepted_count],
            commit.layer.A_log,
            commit.layer.dt_bias,
            commit.old_recurrent_state,
            prefix_mask,
            use_kernel=not commit.layer.training,
        )
        commit.cache[1] = state
        commit.cache.lengths = commit.old_lengths
        commit.cache.left_padding = commit.old_left_padding
        commit.cache.advance(accepted_count)


def _target_forward(
    target: nn.Module,
    inputs: mx.array,
    cache: Optional[list[Any]],
    layer_ids: list[int],
    *,
    input_embeddings: Optional[mx.array] = None,
    gated_delta_commits: Optional[list[_GatedDeltaCommit]] = None,
) -> tuple[mx.array, mx.array]:
    core = _target_core(target)
    hidden_states = input_embeddings if input_embeddings is not None else core.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(core.layers)

    selected = []
    if hasattr(core, "fa_idx") and hasattr(core, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[core.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[core.ssm_idx])
        for idx, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if gated_delta_commits is not None and layer.is_linear:
                hidden_states = _qwen35_decoder_layer_with_commit(
                    layer,
                    hidden_states,
                    mask,
                    layer_cache,
                    gated_delta_commits,
                )
            else:
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if idx in layer_ids:
                selected.append(hidden_states)
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for idx, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if idx in layer_ids:
                selected.append(hidden_states)

    logits = _target_lm_head(target, core.norm(hidden_states))
    return logits, mx.concatenate(selected, axis=-1)


def _trim_cache_to(cache: list[Any], target_size: int) -> None:
    sizes = [entry.size() for entry in cache if hasattr(entry, "size")]
    current = max(sizes) if sizes else target_size
    cache_utils.trim_prompt_cache(cache, max(current - target_size, 0))


@dataclass
class DFlashGenerationResponse:
    text: str
    tokens: list[int]
    accepted: int
    generation_tps: float
    prompt_tokens: int
    peak_memory: float
    finish_reason: Optional[str] = None
    profile: Optional[dict[str, float]] = None


def dflash_generate_step(
    prompt: mx.array,
    target: nn.Module,
    draft: DFlashDraftModel,
    *,
    block_size: Optional[int] = None,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    profile: bool = False,
) -> Generator[tuple[list[int], int, Optional[dict[str, float]]], None, None]:
    block_size = block_size or draft.block_size
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if draft.mask_token_id is None:
        raise ValueError("draft config does not define dflash_config.mask_token_id")

    target_cache = cache_utils.make_prompt_cache(target)
    draft_cache = draft.make_cache()
    target_hidden = None

    prefill_start = _mark(profile)
    while prompt.size > 1:
        logits, target_hidden = _target_forward(target, prompt[:-1][None], target_cache, draft.target_layer_ids)
        mx.eval(logits, target_hidden)
        prompt = prompt[-1:]

    logits, target_hidden = _target_forward(target, prompt[None], target_cache, draft.target_layer_ids)
    pending = _sample(logits[:, -1, :], sampler).item()
    prefill_s = _elapsed(prefill_start, profile)
    pending_emitted = False
    generated = 0

    target_cache_is_trimmable = cache_utils.can_trim_prompt_cache(target_cache)
    target_cache_has_gated_delta_history = _can_commit_gated_delta_history(target, target_cache)

    while generated < max_tokens:
        mask_tail = [draft.mask_token_id] * max(block_size - 1, 0)
        old_target_size = max(c.size() for c in target_cache if hasattr(c, "size"))
        profile_step = None
        checkpoint_s = 0.0
        replay_s = 0.0
        trim_s = 0.0

        block_tokens = mx.array([[pending] + mask_tail], dtype=mx.uint32)
        draft_s = 0.0
        if block_size > 1:
            draft_start = _mark(profile)
            noise_embedding = _target_embedding(target)(block_tokens)
            draft_hidden = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                cache=draft_cache,
            )
            draft_logits = _target_lm_head(target, draft_hidden[:, 1:, :])
            sampled_draft = _sample(draft_logits, sampler)
            block_tokens = mx.concatenate([block_tokens[:, :1], sampled_draft.astype(mx.uint32)], axis=1)
            cache_utils.trim_prompt_cache(draft_cache, block_size)
            draft_s = _elapsed(draft_start, profile)

        verify_start = _mark(profile)
        checkpoint_start = _mark(profile)
        use_gated_delta_history = (not target_cache_is_trimmable) and target_cache_has_gated_delta_history
        target_cache_checkpoint = (
            None if target_cache_is_trimmable or use_gated_delta_history else copy.deepcopy(target_cache)
        )
        checkpoint_s = _elapsed(checkpoint_start, profile)
        gated_delta_commits: Optional[list[_GatedDeltaCommit]] = [] if use_gated_delta_history else None
        verify_logits, verify_hidden = _target_forward(
            target,
            block_tokens,
            target_cache,
            draft.target_layer_ids,
            gated_delta_commits=gated_delta_commits,
        )
        posterior = _sample(verify_logits, sampler).astype(mx.uint32)
        mx.eval(block_tokens, posterior, verify_hidden)
        verify_s = _elapsed(verify_start, profile)

        if block_size > 1:
            matches = (block_tokens[:, 1:] == posterior[:, :-1]).tolist()[0]
            accepted_drafts = 0
            for ok in matches:
                if not ok:
                    break
                accepted_drafts += 1
        else:
            accepted_drafts = 0

        accepted_count = accepted_drafts + 1
        pending = int(posterior[0, accepted_drafts].item())

        if target_cache_is_trimmable:
            trim_start = _mark(profile)
            _trim_cache_to(target_cache, old_target_size + accepted_count)
            trim_s = _elapsed(trim_start, profile)
            target_hidden = verify_hidden[:, :accepted_count, :]
        elif use_gated_delta_history:
            trim_start = _mark(profile)
            _trim_trimmable_cache_entries_to(target_cache, old_target_size + accepted_count)
            _apply_gated_delta_history_commits(gated_delta_commits or [], accepted_count)
            target_hidden = verify_hidden[:, :accepted_count, :]
            mx.eval(target_hidden, *_cache_arrays(target_cache))
            trim_s = _elapsed(trim_start, profile)
        else:
            replay_start = _mark(profile)
            target_cache = target_cache_checkpoint
            replay_tokens = block_tokens[:, :accepted_count]
            _, target_hidden = _target_forward(
                target,
                replay_tokens,
                target_cache,
                draft.target_layer_ids,
            )
            mx.eval(target_hidden)
            replay_s = _elapsed(replay_start, profile)

        accepted_tokens = block_tokens[0, :accepted_count].tolist()
        emitted = accepted_tokens if not pending_emitted else accepted_tokens[1:]
        emitted.append(pending)
        pending_emitted = True

        remaining = max_tokens - generated
        emitted = emitted[:remaining]
        generated += len(emitted)
        if profile:
            profile_step = {
                "prefill_s": prefill_s,
                "cache_checkpoint_s": checkpoint_s,
                "draft_s": draft_s,
                "verify_s": verify_s,
                "cache_trim_s": trim_s,
                "cache_replay_s": replay_s,
                "accepted": float(accepted_count),
                "emitted": float(len(emitted)),
                "block_size": float(block_size),
                "steps": 1.0,
                "target_cache_trimmable": float(target_cache_is_trimmable),
                "block_verify": 1.0,
            }
            prefill_s = 0.0
        yield emitted, accepted_count, profile_step


def stream_generate(
    target: nn.Module,
    draft: DFlashDraftModel,
    tokenizer,
    prompt,
    block_size: Optional[int] = None,
    max_tokens: int = 256,
    *,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    profile: bool = False,
) -> Generator[DFlashGenerationResponse, None, None]:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt, dtype=mx.uint32)

    detokenizer = tokenizer.detokenizer
    prompt_tokens = prompt.size
    emitted_count = 0
    start = time.perf_counter()
    stop_tokens = set(tokenizer.eos_token_ids)
    final_reason = "length"

    for tokens, accepted, profile_step in dflash_generate_step(
        prompt,
        target,
        draft,
        block_size=block_size,
        max_tokens=max_tokens,
        sampler=sampler,
        profile=profile,
    ):
        for token in tokens:
            detokenizer.add_token(token)
        emitted_count += len(tokens)
        finish_reason = "stop" if any(token in stop_tokens for token in tokens) else None
        if finish_reason is None and emitted_count >= max_tokens:
            finish_reason = "length"
        final_reason = finish_reason or final_reason
        yield DFlashGenerationResponse(
            text=detokenizer.last_segment,
            tokens=tokens,
            accepted=accepted,
            generation_tps=emitted_count / max(time.perf_counter() - start, 1e-6),
            prompt_tokens=prompt_tokens,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason,
            profile=profile_step,
        )
        if finish_reason is not None:
            break

    detokenizer.finalize()
    if detokenizer.last_segment:
        yield DFlashGenerationResponse(
            text=detokenizer.last_segment,
            tokens=[],
            accepted=0,
            generation_tps=emitted_count / max(time.perf_counter() - start, 1e-6),
            prompt_tokens=prompt_tokens,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=final_reason,
            profile=None,
        )
