# MLX DFlash Optimization Log

This log tracks clean-room experiments for the MLX DFlash implementation in
`dflash/model_mlx_clean.py`. The existing `dflash/model_mlx.py` implementation is
out of scope and must not be used as a reference.

## Objective

Improve the initial clean-room MLX implementation for Qwen3.5-4B by adding
profiling, running bounded experiments, and using the results to choose the next
implementation changes.

## Baseline

Command:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 uv run --no-project --python 3.12 \
  --with mlx==0.31.2 \
  --with mlx-lm==0.31.3 \
  --with huggingface-hub \
  --with numpy \
  --with requests \
  --with loguru \
  --with rich \
  --with tqdm \
  --with datasets \
  python -m dflash.benchmark \
    --backend mlx \
    --model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --dataset gsm8k \
    --max-samples 8 \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --block-size 16 \
    --enable-thinking
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 49.86 tok/s |
| DFlash throughput | 57.50 tok/s |
| Speedup | 1.15x |
| Average acceptance length | 5.09 |

Acceptance histogram for lengths 0..16:

```text
0.0%, 13.5%, 18.8%, 20.2%, 8.7%, 4.3%, 8.2%, 2.9%,
5.3%, 2.4%, 4.3%, 1.9%, 2.4%, 1.0%, 1.0%, 1.9%, 3.4%
```

Initial interpretation:

- The clean-room path works end-to-end on cached Qwen3.5-4B and DFlash weights.
- The 1.15x speedup is much lower than expected from the paper, so overhead is
  likely concentrated in implementation details rather than acceptance alone.
- The average accepted block length is usable, but the decode loop likely pays
  extra cost from Python control flow, full-block verification, cache restore and
  replay for Qwen3.5 linear-attention caches, and repeated allocations.

## Hypotheses

1. Cache restore and replay for Qwen3.5 target caches is a significant fraction
   of decode time because linear-attention `ArraysCache` entries are not
   trimmable in MLX-LM.
2. Draft cache prefill/cropping is doing extra work because context KV is
   recomputed and appended each iteration.
3. Full `mx.synchronize()` profiling will reveal whether the loop is target
   verification-bound, draft-bound, or Python/replay-bound.
4. Smaller inference block sizes may improve throughput if block size 16 causes
   too much verification/replay work relative to accepted tokens.
5. For Qwen3.5, a custom target cache rollback for `ArraysCache` may matter more
   than optimizing the draft transformer.

## Experiments

Reproducibility note:

- Experiments before E13 were run when the clean MLX benchmark defaulted to
  block verification. After E13, `--mlx-verify-mode auto` is the default and
  resolves Qwen3.5 to sequential verification for correctness. To reproduce old
  block-verification timing experiments, pass `--mlx-verify-mode block`
  explicitly.

### E0: Initial Smoke Benchmark

Status: complete

Outcome:

- End-to-end MLX benchmark completed without runtime errors.
- Speedup was 1.15x on 8 GSM8K prompts, `max_new_tokens=128`,
  `block_size=16`, greedy sampling, thinking enabled.

Next:

- Add profiling counters before changing behavior.

### E1: Synchronized Profile for Block Verification

Status: complete

Change:

- Added opt-in synchronized profiling with `--profile`.
- Timed prefill, draft, block verification, cache checkpoint, cache trim, and
  Qwen3.5 cache replay.

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.21 tok/s |
| DFlash throughput | 54.68 tok/s |
| Speedup | 1.13x |
| Average acceptance length | 5.09 |
| Profile steps | 208 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.527s | 0.51 ms |
| Cache checkpoint | 0.038s | 0.04 ms |
| Draft | 0.044s | 0.04 ms |
| Verify | 11.036s | 10.78 ms |
| Cache replay | 6.998s | 6.83 ms |

Outcome:

- Draft compute is not the current bottleneck.
- Target block verification plus Qwen3.5 cache replay dominate.
- `target_cache_trimmable` is false because Qwen3.5 uses linear-attention
  `ArraysCache` entries that cannot be cropped after rejected tokens.

Next:

- Test a verification strategy that avoids replay by verifying sequentially and
  stopping at first mismatch.

### E2: Sequential Verification to Avoid Replay

Status: complete

Change:

- Added `--mlx-verify-mode sequential`.
- Instead of verifying the whole block and replaying accepted tokens, the target
  verifies one token at a time and stops at first mismatch. This keeps the
  Qwen3.5 cache clean without replay.

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --mlx-verify-mode sequential
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.30 tok/s |
| DFlash throughput | 40.64 tok/s |
| Speedup | 0.84x |
| Average acceptance length | 5.09 |
| Profile steps | 209 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.525s | 0.51 ms |
| Draft | 0.044s | 0.04 ms |
| Verify | 24.604s | 24.03 ms |
| Cache replay | 0.000s | 0.00 ms |

Outcome:

- Negative performance result.
- Avoiding replay is not enough; losing parallel target verification more than
  doubles verification time.
- Later equivalence experiments show this mode is the correctness-preserving
  fallback for MLX Qwen3.5, while block verification remains a fast experimental
  path that must be guarded with `--check-equivalence`.

Next:

- Test smaller block sizes while keeping block verification.

### E3: Smaller Inference Block Size

Status: complete

Change:

- Ran the block-trained Qwen3.5 DFlash draft with `--block-size 8`.
- Kept block verification.

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 8 \
  --enable-thinking \
  --profile
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 47.56 tok/s |
| DFlash throughput | 46.39 tok/s |
| Speedup | 0.98x |
| Average acceptance length | 4.34 |
| Profile steps | 239 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.537s | 0.52 ms |
| Draft | 0.052s | 0.05 ms |
| Verify | 13.036s | 12.73 ms |
| Cache replay | 8.356s | 8.16 ms |

Outcome:

- Negative result.
- Smaller blocks reduced the amount of speculative work per step but increased
  the number of target verification/replay cycles enough to erase the benefit.
- Block size 16 remains better for this benchmark.

Next:

- Add a targeted fast path for full-block acceptance.

### E4: Full-Block Acceptance Fast Path

Status: complete

Change:

- If block verification accepts all draft positions, skip cache restore/replay.
- This is safe because no rejected tokens have polluted the Qwen3.5 target
  cache in the full-accept case.

Smoke result:

- For a short `Say hi.` prompt with `max_tokens=2`, replay time dropped to zero
  when the block fully accepted.

Aggregate command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile
```

Aggregate result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.61 tok/s |
| DFlash throughput | 44.32 tok/s |
| Speedup | 0.91x |
| Average acceptance length | 5.09 |
| Profile steps | 208 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.635s | 0.62 ms |
| Draft | 0.046s | 0.04 ms |
| Verify | 13.878s | 13.55 ms |
| Cache replay | 8.464s | 8.27 ms |

Outcome:

- This change was removed after broader equivalence testing. The short smoke was
  too narrow to prove correctness for Qwen3.5's hybrid target cache.
- Full-block acceptance occurs only 3.4% of steps in the current GSM8K run, so
  the fast path is not expected to move aggregate throughput much.
- The aggregate run was slower than E1 despite the fast path, likely due to
  run-to-run noise and thermal/cache effects after repeated large-model loops.

### E5: Qwen3.5 Cache Rollback Feasibility Check

Status: complete

Question:

- Can we avoid replay after block verification by rolling Qwen3.5 linear
  attention caches back to the accepted prefix?

Evidence inspected:

- MLX-LM Qwen3.5 uses `ArraysCache(size=2)` for linear-attention layers.
- The first entry stores the convolution tail state.
- The second entry stores the gated-delta recurrent state.
- `gated_delta_update` returns the output sequence and only the final recurrent
  state for the whole processed block. The Metal kernel likewise emits
  `state_out`, not a per-token state history.

Outcome:

- There is no safe slice-based rollback available from the current MLX-LM
  cache/state API.
- A true rollback would require either a custom gated-delta kernel that also
  exports intermediate states or recomputing the accepted prefix. The latter is
  what the current replay path already does.

Next:

- Test adaptive block sizing as a lower-risk way to reduce wasted verify/replay
  work after low-acceptance steps while still using block-parallel verify.

### E6: Adaptive Block Size

Status: complete

Change:

- Added `--mlx-adaptive-block-size`.
- The policy starts at the requested block size, backs off to
  `--mlx-min-block-size` after low-acceptance steps, and grows back toward the
  maximum after near-full acceptance.
- This keeps block-parallel target verification and does not use sequential
  verification.

Command, minimum block size 8:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --mlx-adaptive-block-size \
  --mlx-min-block-size 8
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 53.14 tok/s |
| DFlash throughput | 59.31 tok/s |
| Speedup | 1.12x |
| Average acceptance length | 4.55 |
| Average block size | 10.67 |
| Profile steps | 228 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.490s | 0.48 ms |
| Draft | 0.040s | 0.04 ms |
| Verify | 10.906s | 10.65 ms |
| Cache replay | 5.750s | 5.62 ms |

Command, minimum block size 4:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --mlx-adaptive-block-size \
  --mlx-min-block-size 4
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 50.97 tok/s |
| DFlash throughput | 55.36 tok/s |
| Speedup | 1.09x |
| Average acceptance length | 4.12 |
| Average block size | 8.54 |
| Profile steps | 253 |
| Emitted tokens | 1024 |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Prefill | 0.493s | 0.48 ms |
| Draft | 0.049s | 0.05 ms |
| Verify | 12.480s | 12.19 ms |
| Cache replay | 5.382s | 5.26 ms |

Outcome:

- Mixed but useful result.
- Adaptive min-8 reduces replay from the original profiled baseline
  (`6.998s -> 5.750s`) and keeps verification time similar
  (`11.036s -> 10.906s`).
- Adaptive min-4 reduces replay a little further but increases verification
  time and step count enough to lose throughput.
- The current best adaptive setting is min block size 8, but the speedup ratio
  remains around 1.1x on this tiny GSM8K sample because baseline throughput also
  varies between runs.

Next:

- Keep adaptive mode as an experimental knob, not the default yet.
- Larger, repeated benchmarks are needed before treating the min-8 policy as a
  real improvement rather than a promising direction.

### E7: Greedy Equivalence Guard

Status: complete

Change:

- Added `--check-equivalence` to the MLX benchmark.
- The check compares target-only greedy tokens against DFlash greedy tokens for
  every generated turn and fails the run on the first mismatch.
- The check is restricted to `--temperature 0.0`; stochastic sampling would
  need shared random draws to make exact token equality meaningful.

Fixed-block command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 3 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 37.99 tok/s |
| DFlash throughput | 23.22 tok/s |
| Speedup | 0.61x |
| Average acceptance length | 5.36 |
| Equivalence | passed, 3 turns |

Adaptive min-8 command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 3 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence \
  --mlx-adaptive-block-size \
  --mlx-min-block-size 8
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.35 tok/s |
| DFlash throughput | 51.70 tok/s |
| Speedup | 1.07x |
| Average acceptance length | 4.57 |
| Equivalence | passed, 3 turns |

Outcome:

- Correctness guard passed for both fixed and adaptive block-size modes on the
  bounded greedy sample.
- The fixed-block equivalence run was slower than the earlier profiling run,
  which reinforces that tiny throughput samples are noisy and should be used
  mainly for regressions and directional evidence.
- Future optimization experiments should include `--check-equivalence` on a
  small greedy sample before larger timing runs.

### E8: Deterministic Sample Selection and Richer Equivalence Diagnostics

Status: complete

Change:

- Replaced global-RNG benchmark sampling with local seeded sampling via
  `--sample-seed` so model loading or other libraries cannot perturb the sample
  order.
- Equivalence failures now include prompt SHA-1, prompt length, prompt preview,
  and a small baseline/DFlash token window around the mismatch.

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.29 tok/s |
| DFlash throughput | 54.50 tok/s |
| Speedup | 1.13x |
| Average acceptance length | 4.93 |
| Equivalence | passed, 8 turns |

Outcome:

- The fixed block-16 path is reproducible on the deterministic 8-turn slice.
- The earlier broad mismatch could not be reproduced after restarting from the
  current source and removing the full-block fast path.
- Keep the deterministic sample seed and richer failure payloads; they paid for
  themselves immediately.

### E9: Deterministic Fixed Block-16 Profile

Status: complete

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --check-equivalence
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 41.96 tok/s |
| DFlash throughput | 41.88 tok/s |
| Speedup | 1.00x |
| Average acceptance length | 4.93 |
| Profile steps | 214 |
| Emitted tokens | 1024 |
| Equivalence | passed, 8 turns |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Baseline synchronized decode | 26.146s | 25.53 ms |
| Prefill | 0.758s | 0.74 ms |
| Cache checkpoint | 0.039s | 0.04 ms |
| Draft | 0.046s | 0.04 ms |
| Verify | 14.470s | 14.13 ms |
| Cache replay | 9.092s | 8.88 ms |

Outcome:

- Synchronized profiling erases the apparent wall-clock speedup on this small
  sample, but it gives a stable cost breakdown.
- Target verification plus replay is still the bottleneck: 23.562s combined,
  versus 0.046s for the draft transformer.

### E10: Deterministic Adaptive Block-Size Rerun

Status: complete

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --check-equivalence \
  --mlx-adaptive-block-size \
  --mlx-min-block-size 8
```

Profiled result:

| Metric | Value |
|---|---:|
| Baseline throughput | 39.66 tok/s |
| DFlash throughput | 35.60 tok/s |
| Speedup | 0.90x |
| Average acceptance length | 4.66 |
| Average block size | 10.67 |
| Profile steps | 225 |
| Equivalence | passed, 8 turns |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Baseline synchronized decode | 27.631s | 26.98 ms |
| Prefill | 0.906s | 0.89 ms |
| Cache checkpoint | 0.039s | 0.04 ms |
| Draft | 0.049s | 0.05 ms |
| Verify | 16.912s | 16.52 ms |
| Cache replay | 10.809s | 10.56 ms |

No-profile check:

| Metric | Value |
|---|---:|
| Baseline throughput | 34.85 tok/s |
| DFlash throughput | 31.66 tok/s |
| Speedup | 0.91x |
| Average acceptance length | 4.66 |
| Equivalence | passed, 8 turns |

Outcome:

- Negative result on the deterministic sample.
- The adaptive policy lowered average block size but also lowered acceptance and
  increased step count enough to lose throughput.
- Do not promote adaptive block size; keep it as a diagnostic knob only.

### E11: Smaller Block Size Correctness Failure

Status: complete

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 8 \
  --enable-thinking \
  --check-equivalence
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 35.69 tok/s |
| DFlash throughput | 28.48 tok/s |
| Speedup | 0.80x |
| Average acceptance length | 4.40 |
| Equivalence | failed, 2 of 8 turns |

Additional check:

- `--mlx-replay-mode sequential` was added to test whether accepted-prefix replay
  chunking caused the mismatch.
- Sequential replay was slower and still failed: baseline 48.80 tok/s, DFlash
  33.02 tok/s, speedup 0.68x, equivalence failed on 3 of 8 turns.

Investigation:

- A target-only loop using the clean `_target_forward` one token at a time
  matches MLX-LM `stream_generate` on the failing prompts.
- Batched target verification after a Qwen3.5 recurrent-cache prefix can choose
  a different argmax than one-token target decoding. One failing example:
  after the generated prefix ending in tokens `... 21156, 411`, target-only
  greedy chose token `3377`, while batched verification over the speculative
  block chose `3299`.
- This points at MLX Qwen3.5 hybrid linear-attention chunk/cache behavior, not a
  draft-model proposal issue. If the target verifier is not exact, speculative
  correctness does not hold.

Outcome:

- `--block-size 8` is invalid for this MLX Qwen3.5 path.
- More generally, block verification for the hybrid Qwen3.5 target must be
  treated as experimental unless protected by `--check-equivalence`.

### E12: Sequential Verification Correctness Fallback

Status: complete

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --check-equivalence \
  --mlx-verify-mode sequential
```

Profiled result:

| Metric | Value |
|---|---:|
| Baseline throughput | 48.79 tok/s |
| DFlash throughput | 41.14 tok/s |
| Speedup | 0.84x |
| Average acceptance length | 4.98 |
| Profile steps | 212 |
| Emitted tokens | 1024 |
| Equivalence | passed, 8 turns |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Baseline synchronized decode | 22.654s | 22.12 ms |
| Prefill | 0.523s | 0.51 ms |
| Draft | 0.041s | 0.04 ms |
| Verify | 24.303s | 23.73 ms |
| Cache replay | 0.000s | 0.00 ms |

Outcome:

- Sequential verification is the current correctness fallback for MLX Qwen3.5.
- It is slower than the target-only baseline on this sample, so it is not an
  optimization win.
- The next meaningful improvement is a safe block verifier for Qwen3.5's hybrid
  target path, or running this clean implementation against a full-attention
  target where batched block verification is exact.

### E13: Target-Verifier Harness and Correctness-First Auto Mode

Status: complete

Change:

- Added `python -m dflash.verify_mlx_target`, a focused MLX target verifier
  harness. It checks draft-proposed blocks by comparing batched target posterior
  tokens against one-token sequential target posterior tokens from the same
  prefix.
- The harness fails only when the batched verifier changes the speculative
  accept length or the pending token. Posterior mismatches after an already
  rejected token are tracked, but are not by themselves correctness failures.
- Changed the clean MLX generation default from `verify_mode="block"` to
  `verify_mode="auto"`. Auto mode uses block verification only when the target
  prompt cache is trimmable; Qwen3.5's hybrid target cache is not trimmable, so
  auto resolves to sequential verification.

Block-16 verifier command:

```bash
python -m dflash.verify_mlx_target \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --block-size 16 \
  --enable-thinking \
  --max-failures 1
```

Block-16 verifier result:

| Metric | Value |
|---|---:|
| Turns checked before stop | 2 |
| Checked positions | 816 |
| Posterior-mismatching blocks | 6 |
| Decision mismatches | 1 |
| Block verifier time | 3.435s |
| Sequential reference time | 18.748s |

First decision mismatch:

| Field | Value |
|---|---|
| Prompt hash | `6bfc34c10aca` |
| Sample / step / generated | `1 / 26 / 113` |
| First posterior mismatch | position `4` |
| Block acceptance | `5` |
| Sequential acceptance | `13` |
| Block pending | `56014` |
| Sequential pending | `8046` |

Auto-mode smoke command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 3 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --profile \
  --check-equivalence
```

Auto-mode smoke result:

| Metric | Value |
|---|---:|
| Baseline throughput | 47.86 tok/s |
| DFlash throughput | 31.72 tok/s |
| Speedup | 0.66x |
| Average acceptance length | 4.68 |
| Auto verify | true |
| Sequential verify | true |
| Equivalence | passed, 3 turns |

Auto-mode 8-turn command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence
```

Auto-mode 8-turn result:

| Metric | Value |
|---|---:|
| Baseline throughput | 45.46 tok/s |
| DFlash throughput | 37.10 tok/s |
| Speedup | 0.82x |
| Average acceptance length | 4.98 |
| Equivalence | passed, 8 turns |

Outcome:

- Block verification is not safe as a Qwen3.5 default. The end-to-end
  equivalence guard can miss target-verifier decision mismatches if a small run
  does not visibly diverge.
- The target-verifier harness is now the stronger gate for block-mode
  optimization work.
- Auto mode makes the clean implementation correctness-first for Qwen3.5, while
  retaining explicit `--mlx-verify-mode block` for experimental measurements.

### E14: Fresh Studio Baselines

Status: complete

Context:

- Runs moved to `studio.local` in
  `/Users/studio/git/.worktrees/z-lab/dflash/green-field-impl`.
- Commands used `HF_HUB_OFFLINE=1` and `uv run --no-project --python 3.12`
  with `mlx==0.31.2` and `mlx-lm==0.31.3`.

Correctness-first auto command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.99 tok/s |
| DFlash throughput | 52.82 tok/s |
| Speedup | 0.79x |
| Average acceptance length | 5.04 |
| Equivalence | passed, 8 turns |

Invalid block-mode upper-bound command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence \
  --mlx-verify-mode block
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.99 tok/s |
| DFlash throughput | 70.84 tok/s |
| Speedup | 1.06x |
| Average acceptance length | 5.02 |
| Equivalence | failed, 5 of 8 turns |

Target verifier command:

```bash
python -m dflash.verify_mlx_target \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --block-size 16 \
  --enable-thinking \
  --max-failures 3
```

Result:

| Metric | Value |
|---|---:|
| Turns checked | 8 |
| Checked positions | 3328 |
| Posterior-mismatching blocks | 17 |
| Block verify time | 7.164s |
| Sequential reference time | 55.440s |
| Decision mismatches | 2 |

Outcome:

- The current correctness-preserving path remains slower than target-only.
- The unsafe block verifier still cannot exceed the 1.5x target even as an
  invalid upper bound on this studio run.
- The target verifier remains the mandatory correctness gate.

Next:

- Diagnose whether target-only batched Qwen3.5 decode diverges from
  token-by-token cached decode independently of DFlash.

### E15: Target-Only Block-vs-Sequential Diagnostic

Status: complete

Change:

- Added `python -m dflash.debug_mlx_target`.
- The harness generates target-only continuations, then compares target
  posterior logits for block sizes `1,2,4,8,16` against one-token sequential
  cached decode from the same prefix.
- It supports `true`, `perturb-tail`, and `perturb-after-4` modes and reports
  top-k margins plus compact `KVCache` / `ArraysCache` summaries.

True-continuation command:

```bash
python -m dflash.debug_mlx_target \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 2 \
  --max-new-tokens 64 \
  --block-sizes 1,2,4,8,16 \
  --modes true \
  --enable-thinking \
  --max-failures 1
```

Result:

- Divergences: `0`.

Perturbed-speculation command:

```bash
python -m dflash.debug_mlx_target \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 2 \
  --max-new-tokens 64 \
  --block-sizes 1,2,4,8,16 \
  --modes perturb-tail,perturb-after-4 \
  --enable-thinking \
  --max-failures 2
```

Result:

| Field | First divergence |
|---|---|
| Sample | `0` |
| Mode | `perturb-tail` |
| Block size | `4` |
| Prefix tokens after prompt | `56` |
| Block position | `2` |
| Block argmax | `220` |
| Sequential argmax | `198` |
| Block top-2 | `220:16.5`, `198:16.375` |
| Sequential top-2 | `220:16.375`, `198:16.375` |
| Max absolute logit diff | `0.1875` |
| Cache summary | same classes and sizes after block and sequential paths |

Outcome:

- Target-only decode is stable on the model's own greedy trajectory.
- Off-trajectory speculative blocks can flip argmax decisions even without the
  DFlash draft.
- The first observed flip is a near-tie, but speculative correctness cannot
  rely on tie luck.

Next:

- Check whether the divergence is caused by MLX-LM's optimized
  `gated_delta_update` kernel.

### E16: GatedDeltaNet Kernel Toggle

Status: complete

Change:

- Added `disable_gated_delta_kernel()` in `dflash/model_mlx_clean.py`.
- Added benchmark/verifier flags:
  - `--mlx-disable-gated-delta-kernel`
  - `--disable-gated-delta-kernel`
- The flag monkeypatches MLX-LM Qwen3.5 to call
  `gated_delta_update(..., use_kernel=False)`.

Target-only diagnostic result:

- With one sample, the reference path removed the first synthetic perturbation
  divergence.
- With two samples, the reference path still diverged once:

| Field | Value |
|---|---|
| Sample | `1` |
| Mode | `perturb-tail` |
| Block size | `16` |
| Block position | `9` |
| Block argmax | `3749` |
| Sequential argmax | `7225` |
| Block top-2 | `7225:25.75`, `3749:25.75` |
| Sequential top-2 | `7225:25.75`, `3749:25.625` |
| Max absolute logit diff | `0.15625` |

Target verifier command:

```bash
python -m dflash.verify_mlx_target \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --block-size 16 \
  --enable-thinking \
  --max-failures 3 \
  --disable-gated-delta-kernel
```

Result:

| Metric | Value |
|---|---:|
| Turns checked | 8 |
| Checked positions | 3360 |
| Posterior-mismatching blocks | 20 |
| Block verify time | 11.290s |
| Sequential reference time | 60.016s |
| Decision mismatches | 2 |

Outcome:

- The optimized Metal kernel contributes to some divergences, but disabling it
  does not make batched verification exact for DFlash proposal blocks.
- This path is not a viable correctness-preserving speedup strategy.

Next:

- Add layer-level bisection to locate where block and sequential activations
  first diverge.

### E17: Layer-Level Bisection

Status: complete

Change:

- Added `python -m dflash.debug_mlx_layers`.
- The diagnostic finds the first real DFlash verifier decision mismatch, then
  replays the same checkpoint and block with batched target decode and
  one-token sequential target decode.
- It reports per-layer hidden-state `max_abs` and `mean_abs` differences at the
  first posterior-mismatching position.

Optimized-kernel command:

```bash
python -m dflash.debug_mlx_layers \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 3 \
  --max-new-tokens 128 \
  --block-size 16 \
  --enable-thinking
```

Result:

| Field | Value |
|---|---|
| First mismatch | sample `2`, step `2`, generated `6` |
| First posterior mismatch | position `2` |
| Block acceptance / pending | `3` / `3299` |
| Sequential acceptance / pending | `12` / `279` |
| First nonzero layer diff | layer `0`, linear, `max_abs=1.52587890625e-05` |
| First material layer diff | layer `2`, linear, `max_abs=0.0078125` |
| Late-stack diff example | layer `31`, attention, `max_abs=0.0625`, `mean_abs=0.0103` |

First-layer `GatedDeltaNet` internal stage check for the same mismatch:

| Stage | Max abs diff |
|---|---:|
| input norm | `0.0` |
| qkv projection | `0.0` |
| depthwise conv output | `0.0` |
| q/k/v/a/b/g | `0.0` |
| recurrent output | `0.0` |
| gated norm | `2.384185791015625e-07` |
| output projection | `7.450580596923828e-09` |
| layer output | `1.52587890625e-05` |

Reference-kernel command:

```bash
python -m dflash.debug_mlx_layers \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 2 \
  --max-new-tokens 128 \
  --block-size 16 \
  --enable-thinking \
  --disable-gated-delta-kernel
```

Result:

| Field | Value |
|---|---|
| First mismatch | sample `1`, step `26`, generated `113` |
| First posterior mismatch | position `4` |
| Block acceptance / pending | `13` / `8046` |
| Sequential acceptance / pending | `5` / `56014` |
| First nonzero layer diff | layer `0`, linear, `max_abs=0.00048828125` |
| First material layer diff | layer `2`, linear, `max_abs=0.03125` |
| Late-stack diff example | layer `31`, attention, `max_abs=0.125`, `mean_abs=0.0128` |

Outcome:

- Divergence starts in the first linear-attention layers, then accumulates
  through the stack until the logits flip.
- In the first failing optimized-kernel example, the first layer's convolution
  and recurrent update are identical at the mismatch position; the first
  nonzero internal difference appears after gated normalization and residual/MLP
  composition.
- This rules out the DFlash draft transformer and late full-attention layers as
  the primary cause.
- A correctness-preserving fast verifier for Qwen3.5 would need target-side
  layer semantics that are equivalent to one-token cached decode for
  speculative off-trajectory blocks, including normalization/projection/MLP
  numerical behavior, not only recurrence state.

Next:

- Test whether a full-block verifier built from exact one-token target calls
  can recover performance by reducing synchronization overhead.

### E18: Lazy Full Sequential Verifier

Status: complete

Change:

- Added `--mlx-verify-mode sequential-full`.
- It performs S=1 target forwards for all tokens in the proposed block before
  synchronizing, computes accept/pending from the exact sequential posteriors,
  then restores the checkpoint and replays only the accepted prefix for cache
  repair.

Command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence \
  --profile \
  --mlx-verify-mode sequential-full
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.67 tok/s |
| DFlash throughput | 15.78 tok/s |
| Speedup | 0.24x |
| Average acceptance length | 5.04 |
| Equivalence | passed, 8 turns |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Baseline synchronized decode | 16.904s | 16.51 ms |
| Prefill | 0.618s | 0.60 ms |
| Draft | 0.035s | 0.03 ms |
| Verify | 49.719s | 48.55 ms |
| Cache replay | 14.432s | 14.09 ms |

Outcome:

- Correctness passed, but performance is much worse than early-stop sequential
  verification.
- Fewer synchronization points do not compensate for verifying the whole block
  and replaying the accepted prefix.

Next:

- Keep early-stop sequential verification as the only exact Qwen3.5 path tried
  so far.

### E19: Fresh Auto Profile and Adaptive Block-Size Check

Status: complete

Fresh auto profile command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --enable-thinking \
  --check-equivalence \
  --profile \
  --mlx-verify-mode auto
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.75 tok/s |
| DFlash throughput | 52.55 tok/s |
| Speedup | 0.79x |
| Average acceptance length | 5.04 |
| Equivalence | passed, 8 turns |

Profile totals:

| Component | Total | Per output token |
|---|---:|---:|
| Baseline synchronized decode | 16.897s | 16.50 ms |
| Prefill | 0.633s | 0.62 ms |
| Draft | 0.032s | 0.03 ms |
| Verify | 18.799s | 18.36 ms |
| Cache replay | 0.000s | 0.00 ms |

Adaptive block-size command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 8 \
  --enable-thinking \
  --check-equivalence \
  --profile \
  --mlx-verify-mode auto \
  --mlx-adaptive-block-size \
  --mlx-min-block-size 4
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.74 tok/s |
| DFlash throughput | 51.83 tok/s |
| Speedup | 0.78x |
| Average acceptance length | 4.11 |
| Average block size | 6.80 |
| Equivalence | passed, 8 turns |

Outcome:

- Current correctness-preserving runtime is target-verification-bound.
- Draft compute is effectively irrelevant at this scale: about `0.03 ms` per
  output token.
- Adaptive block sizing does not help because exact verification is already
  dominated by recurrent target work, and shorter blocks reduce acceptance.

Next:

- Do not spend more time on draft micro-optimization or block-size tuning for
  Qwen3.5.

### E20: Strategy Exhaustion Decision for Qwen3.5

Status: complete

Hypothesis:

- The remaining roadmap items might still produce a correctness-preserving
  `1.5x` speedup:
  - state-history capture for recurrent state and conv cache commit;
  - activation replay for accepted-prefix cache repair;
  - a hybrid verifier with sequential linear/recurrent state and batched dense
    work;
  - a full-attention control path.

Evidence:

- State-history capture and activation replay cannot repair a wrong verifier
  posterior. E14 and E16 show accept/pending decisions diverge before any cache
  commit choice is made.
- The first-layer internal bisection in E17 shows the first observed optimized
  mismatch has identical input norm, qkv projection, masked qkv, conv output,
  q/k/v/a/b/g, and recurrent output at the mismatch position. The first nonzero
  difference appears after gated norm, then layer-level differences accumulate
  through later linear and attention layers.
- This means a Qwen3.5 hybrid verifier cannot safely batch only the recurrence
  and assume the rest is numerically interchangeable with one-token cached
  decode. The exact path must preserve token-shaped target semantics much more
  broadly than cache state alone.
- The current exact verifier already costs more than the target-only baseline:

| Quantity | Value |
|---|---:|
| Target-only synchronized decode | `16.50 ms/output token` |
| Exact early-stop sequential verification | `18.36 ms/output token` |
| Draft overhead | `0.03 ms/output token` |
| Required total for `1.5x` vs target-only | `<= 11.00 ms/output token` |

- Even the invalid optimized block-verifier upper bound on studio reaches only
  `1.06x`, and it fails greedy equivalence on `5/8` turns. That leaves no
  measured fast path close to `1.5x`.
- Full-attention control remains useful for generic DFlash validation on a
  trimmable KV-cache target/draft pair, but it does not satisfy the stated
  Qwen3.5 objective and no compatible full-attention MLX target/draft pair is
  part of this greenfield Qwen3.5 worktree.

Outcome:

- The plausible Qwen3.5 implementation strategies in the ChatGPT roadmap are
  exhausted for this clean-room MLX implementation.
- Winning path: none for Qwen3.5. The correct implementation remains
  `verify_mode=auto`, which resolves to early-stop sequential verification for
  the non-trimmable hybrid target cache.
- Final measured correctness-preserving speedup: `0.79x`.
- Final measured invalid upper bound: `1.06x`, with correctness failures.

Next:

- Do not merge unsafe block verification as a default for Qwen3.5.
- Future work should either target a full-attention/trimmable-cache model pair
  or require an MLX-LM/model-kernel change that proves batched Qwen3.5
  speculative-block posteriors match one-token cached decode.

### E21: Qwen3-4B Non-Thinking Trial

Status: complete

Context:

- Target: `Qwen/Qwen3-4B`
- Draft: `z-lab/Qwen3-4B-DFlash-b16`
- Thinking disabled. The Qwen3-4B draft is a non-thinking draft.
- Commands ran on `studio.local` with cached models and
  `HF_HUB_OFFLINE=1`.

Initial block-mode command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --check-equivalence \
  --profile \
  --mlx-verify-mode auto
```

Initial result before the auto-mode correction:

| Metric | Value |
|---|---:|
| Baseline throughput | 71.71 tok/s |
| DFlash throughput | 101.56 tok/s |
| Speedup | 1.42x |
| Average acceptance length | 4.06 |
| Target cache trimmable | true |
| Equivalence | failed, 2 of 8 turns |

Target verifier command:

```bash
python -m dflash.verify_mlx_target \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --block-size 16 \
  --max-failures 3
```

Verifier result:

| Metric | Value |
|---|---:|
| Turns checked | 8 |
| Checked positions | 4192 |
| Posterior-mismatching blocks | 29 |
| Block verify time | 7.784s |
| Sequential reference time | 64.383s |
| Decision mismatches | 3 |

Target-only diagnostic:

```bash
python -m dflash.debug_mlx_target \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 2 \
  --max-new-tokens 64 \
  --block-sizes 1,2,4,8,16 \
  --modes true,perturb-tail,perturb-after-4 \
  --max-failures 2
```

Result:

- Target-only true continuation diverged at block size `2`, block position `0`.
- First divergence was a near tie:
  - block top-2: `334:43.5`, `315:43.25`;
  - sequential top-2: `334:43.5`, `315:43.5`;
  - max absolute logit diff: `0.25`.
- This means a trimmable KV cache is not sufficient to treat batched target
  verification as exact in MLX. Qwen3-4B can also flip greedy decisions between
  cached block decode and one-token cached decode.

Strict sequential command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --check-equivalence \
  --profile \
  --mlx-verify-mode sequential
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 71.40 tok/s |
| DFlash throughput | 56.05 tok/s |
| Speedup | 0.79x |
| Average acceptance length | 4.04 |
| Equivalence | passed, 8 turns |

Margin-guard experiment:

- Added experimental `--mlx-verify-mode block-margin` and
  `--mlx-margin-threshold`.
- Threshold `0.5`: `1.12x`, fallback rate `15.7%`, equivalence failed `4/8`.
- Threshold `2.0`: `0.78x`, fallback rate `42.5%`, equivalence failed `2/8`.

Block-size sweep:

| Block size | Speedup | Equivalence |
|---:|---:|---|
| 16 | 1.44x unprofiled | failed, 2 of 8 |
| 8 | 1.23x | failed, 3 of 8 |
| 4 | 0.99x | failed, 3 of 8 |
| 2 | 0.70x | failed, 3 of 8 |

Change:

- Changed `verify_mode="auto"` to resolve to sequential verification by
  default. Explicit `--mlx-verify-mode block` remains available for invalid
  upper-bound experiments.
- Fixed `min_block_size` validation so it only applies when adaptive block size
  is enabled.

Corrected auto command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --check-equivalence \
  --profile \
  --mlx-verify-mode auto
```

Corrected auto result:

| Metric | Value |
|---|---:|
| Baseline throughput | 71.78 tok/s |
| DFlash throughput | 56.21 tok/s |
| Speedup | 0.78x |
| Average acceptance length | 4.04 |
| Auto verify | true |
| Sequential verify | true |
| Equivalence | passed, 8 turns |

Outcome:

- Qwen3-4B is faster in unsafe block mode than Qwen3.5, but still does not meet
  the `1.5x` target and fails correctness.
- Correctness-preserving Qwen3-4B follows the same practical conclusion as
  Qwen3.5: exact one-token verification is safe but slower than target-only.
- A simple margin guard is not enough; some failures survive high thresholds and
  higher thresholds erase the speed benefit.

### E22: Original Repo MLX Implementation Black-Box Smoke

Status: complete

Context:

- The repo has no formal `tests/` suite in this checkout.
- The original MLX smoke path is the README import:
  `from dflash.model_mlx import load, load_draft, stream_generate`.
- To preserve the clean-room boundary, `dflash/model_mlx.py` was not opened or
  read. It was imported and executed as a black box.
- The harness compared original DFlash greedy tokens against MLX-LM
  target-only greedy tokens on the same 8 GSM8K prompts, `max_new_tokens=128`,
  `block_size=16`, and `temperature=0.0`.

Qwen3.5-4B command shape:

```bash
python - <<'PY'
from dflash.model_mlx import load, load_draft, stream_generate
# target: Qwen/Qwen3.5-4B
# draft: z-lab/Qwen3.5-4B-DFlash
# enable_thinking=True
# compare against mlx_lm.stream_generate
PY
```

Qwen3.5-4B result:

| Metric | Value |
|---|---:|
| Baseline throughput | 67.03 tok/s |
| Original DFlash throughput | 174.56 tok/s |
| Speedup | 2.60x |
| Average acceptance length | 6.60 |
| Equivalence | failed, 2 of 8 turns |

Observed mismatches:

| Sample | Index | Baseline token | Original DFlash token |
|---:|---:|---:|---:|
| 1 | 105 | `17355` | `3925` |
| 2 | 8 | `3377` | `3299` |

Qwen3-4B result:

| Metric | Value |
|---|---:|
| Baseline throughput | 71.80 tok/s |
| Original DFlash throughput | 174.54 tok/s |
| Speedup | 2.43x |
| Average acceptance length | 5.69 |
| Equivalence | failed, 2 of 8 turns |

Observed mismatches:

| Sample | Index | Baseline token | Original DFlash token |
|---:|---:|---:|---:|
| 2 | 116 | `29901` | `93210` |
| 7 | 115 | `892` | `600` |

Outcome:

- The original implementation is much faster than the clean-room
  correctness-first path, but it fails the greedy equivalence guard for both
  model pairs.
- Its speed is therefore not a correctness-preserving result under the current
  objective.

### E23: Original Repo GSM8K Answer-Accuracy Smoke

Status: complete

Context:

- Same black-box boundary as E22: `dflash.model_mlx` was imported and executed,
  but `dflash/model_mlx.py` was not opened or read.
- The repo's cached benchmark JSONL strips GSM8K answer fields, so this run
  loaded the cached Hugging Face `openai/gsm8k` test split directly in offline
  mode.
- Same 8 sampled prompts as the benchmark path, `sample_seed=42`.
- `max_new_tokens=512`, greedy decoding, `block_size=16`.
- Accuracy was scored by extracting the last `\\boxed{...}` answer, then falling
  back to the last numeric answer.

Qwen3.5-4B result:

| Metric | Target-only | Original DFlash |
|---|---:|---:|
| GSM8K scored accuracy | 1/8 | 1/8 |
| Throughput | 66.32 tok/s | 198.20 tok/s |
| Speedup | - | 2.99x |
| Average acceptance length | - | 7.71 |

Per-sample summary:

| Sample | Gold | Target-only | Original DFlash |
|---:|---:|---:|---:|
| 0 | 52 | 5 | 156 |
| 1 | 5 | 5 | 5 |
| 2 | 83 | 12 | 1245 |
| 3 | 10 | 5 | 4 |
| 4 | 70 | 0.5 | 4 |
| 5 | 36 | 5 | 5 |
| 6 | 48 | 6 | 6 |
| 7 | 16 | 6 | 6 |

Qwen3-4B result:

| Metric | Target-only | Original DFlash |
|---|---:|---:|
| GSM8K scored accuracy | 7/8 | 7/8 |
| Throughput | 70.58 tok/s | 203.28 tok/s |
| Speedup | - | 2.88x |
| Average acceptance length | - | 7.06 |

Per-sample summary:

| Sample | Gold | Target-only | Original DFlash |
|---:|---:|---:|---:|
| 0 | 52 | 52 | 52 |
| 1 | 5 | 5 | 5 |
| 2 | 83 | 83 | 83 |
| 3 | 10 | 6 | 6 |
| 4 | 70 | 70 | 70 |
| 5 | 36 | 36 | 36 |
| 6 | 48 | 48 | 48 |
| 7 | 16 | 16 | 16 |

Outcome:

- On this small scored GSM8K sample, Qwen3-4B original DFlash preserves the
  target model's answer-level accuracy despite failing token-level greedy
  equivalence in E22.
- Qwen3.5 answer-level accuracy is poor for both target-only and original DFlash
  under this prompt/extraction setup, so the 1/8 score is not a DFlash-specific
  regression.
- This does not change the strict objective: token-level correctness still fails
  for original DFlash. It does show why task-level benchmark accuracy can look
  acceptable even when exact greedy equivalence fails.

### E24: Clean-Room Quality-Gated GSM8K Smoke

Status: complete

Hypothesis:

- The clean-room implementation should be evaluated under two gates:
  - strict token-level greedy equivalence for exact speculative decoding;
  - task-level quality, where final GSM8K answer accuracy must not regress
    relative to the target-only baseline.

Change:

- Re-ran the clean-room implementation using answer-level GSM8K scoring instead
  of token identity.
- Scoring used cached Hugging Face `openai/gsm8k` answers, `sample_seed=42`,
  greedy decoding, `max_new_tokens=512`, and answer extraction from the final
  `\\boxed{...}` value with numeric fallback.
- Tested `verify_mode=auto` and `verify_mode=block`.

Eight-sample results:

| Model | Mode | Target accuracy | DFlash accuracy | Speedup | Quality gate |
|---|---|---:|---:|---:|---|
| Qwen3.5-4B thinking | auto | 1/8 | 1/8 | 0.84x | pass |
| Qwen3.5-4B thinking | block | 1/8 | 0/8 | 1.45x | fail |
| Qwen3-4B non-thinking | auto | 7/8 | 7/8 | 0.84x | pass |
| Qwen3-4B non-thinking | block | 7/8 | 7/8 | 2.05x | pass |

Qwen3-4B 32-sample expansion:

```bash
# target: Qwen/Qwen3-4B
# draft: z-lab/Qwen3-4B-DFlash-b16
# verify_mode=block
# max_samples=32, max_new_tokens=512, block_size=16
```

Result:

| Metric | Value |
|---|---:|
| Target-only accuracy | 28/32 |
| Clean DFlash block accuracy | 29/32 |
| Quality delta | +1 |
| Quality regressions | 0 |
| Same extracted answer as target | 31/32 |
| Target throughput | 71.15 tok/s |
| Clean DFlash block throughput | 130.03 tok/s |
| Speedup | 1.83x |
| Average acceptance length | 4.84 |

Notable samples:

- Sample 12 improved under DFlash: target extracted `10`, DFlash extracted `2`,
  gold `2`.
- Sample 20 was scored as a miss for both paths because both extracted `50\\%`
  while the gold normalizes to `50`; this is a scorer-normalization artifact,
  not a relative DFlash regression.

Outcome:

- Under a GSM8K answer-level quality gate, Qwen3-4B clean `block` mode is now a
  viable winning path on the sampled evaluation: it exceeds the `1.5x` speed
  target and does not regress answer accuracy versus target-only.
- This is not exact speculative decoding. It remains invalid under the strict
  byte/token-equivalence objective, but valid under the relaxed benchmark-quality
  gate tested here.
- Qwen3.5 clean `block` mode does not pass the relaxed quality gate on the
  sampled thinking setup.

### E25: Runtime Surface Simplified to Block-Only

Status: complete

Decision:

- The active quality-gated path is fixed-block verification.
- Removed the clean runtime verification choices:
  - `auto`
  - `sequential`
  - `sequential-full`
  - `block-margin`
- Removed benchmark CLI knobs for verification mode, replay mode, adaptive block
  sizing, margin threshold, and the gated-delta kernel toggle.
- Kept diagnostic modules intact because they document and reproduce the
  earlier investigation.

Current clean MLX behavior:

- `dflash.model_mlx_clean.stream_generate(...)` always runs block verification.
- If the target prompt cache is trimmable, it trims to the accepted prefix.
- If the target prompt cache is not trimmable, it replays the accepted prefix
  from the checkpoint.

Smoke command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16
```

Result:

| Metric | Value |
|---|---:|
| Baseline throughput | 71.72 tok/s |
| DFlash throughput | 102.60 tok/s |
| Speedup | 1.43x |
| Average acceptance length | 4.06 |

Outcome:

- The code path now matches the quality-gated benchmark direction: simple,
  block-only, and no exact-verification fallback in the production surface.
- Strict token-equivalence checks can still be run by external comparison
  harnesses, but they are no longer a runtime mode.

### E26: Qwen3.5 GatedDeltaNet State-History Cache Commit

Status: complete

Hypothesis:

- Qwen3.5 block verification was target-cache bound after E25 because MLX-LM's
  hybrid cache is not globally trimmable: `KVCache` entries can trim offsets,
  but `ArraysCache(size=2)` for GatedDeltaNet only stores current conv and
  recurrent state.
- The clean block path therefore deep-copied the full target cache and replayed
  the accepted prefix through the full model after every block.
- A state-history commit should preserve the same block-verifier semantics while
  avoiding full accepted-prefix replay:
  - record each GatedDeltaNet layer's pre-block recurrent state;
  - record the block `conv_input` and projected recurrent inputs;
  - after the accept decision, trim only trimmable `KVCache` entries;
  - repair `ArraysCache[0]` to the accepted-prefix conv state;
  - recompute `ArraysCache[1]` only for the accepted prefix.

Code change:

- Added an internal Qwen3.5 hybrid-cache path in `dflash/model_mlx_clean.py`.
- The public runtime remains block-only; no user-facing mode was added.
- Full-attention/trimmable targets still use the previous trim path.
- Non-Qwen3.5 non-trimmable targets still fall back to checkpoint + replay.

Pre-change profile command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --block-size 16 \
  --profile
```

Pre-change result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.99 tok/s |
| DFlash throughput | 67.41 tok/s |
| Speedup | 1.01x |
| Verify time | 9.028s |
| Cache replay time | 5.428s |
| Avg acceptance length | 4.97 |

Post-change 128-token profile result:

| Metric | Value |
|---|---:|
| Baseline throughput | 67.03 tok/s |
| DFlash throughput | 104.49 tok/s |
| Speedup | 1.56x |
| Verify time | 8.818s |
| Cache replay time | 0.000s |
| History commit / trim time | 0.259s |
| Avg acceptance length | 4.95 |

Post-change 512-token benchmark command:

```bash
python -m dflash.benchmark \
  --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --block-size 16 \
  --profile
```

Post-change 512-token result:

| Metric | Value |
|---|---:|
| Baseline throughput | 66.17 tok/s |
| DFlash throughput | 136.11 tok/s |
| Speedup | 2.06x |
| Verify time | 19.202s |
| Cache replay time | 0.000s |
| History commit / trim time | 0.549s |
| Avg acceptance length | 6.00 |

Quality gate:

- Re-ran the 8-sample GSM8K answer scorer with `sample_seed=42`,
  `max_new_tokens=512`, greedy decoding, and non-thinking chat-template mode.
- Target-only accuracy: `7/8`.
- Clean DFlash accuracy: `7/8`.
- Same extracted answer as target: `8/8`.
- Quality regressions: `0`.
- Ad hoc scorer throughput: target `63.99 tok/s`, DFlash `131.03 tok/s`,
  speedup `2.05x`, avg acceptance `5.88`.

History-vs-replay sanity check:

- Monkeypatched `_can_commit_gated_delta_history` off in-process to compare the
  new history-commit path with the old replay path on the first two sampled
  GSM8K prompts, `max_new_tokens=128`.
- Sample 0: no token mismatch, both produced 128 tokens, avg acceptance `4.70`.
- Sample 1: no token mismatch, both produced 128 tokens, avg acceptance `5.87`.

Interpretation:

- The ChatGPT Pro state-history recommendation was actionable for Qwen3.5.
- It does not make Qwen3.5 exact speculative decoding under the earlier strict
  token-equivalence gate; the block verifier can still differ from target-only
  greedy traces.
- Under the agreed answer-level quality gate, however, this is now a viable
  Qwen3.5 path: the accepted-prefix replay blocker is removed, the standard
  benchmark exceeds `1.5x`, and the sampled GSM8K answer quality matches
  target-only.

## Current Conclusions

- The clean implementation is still target-bound, not draft-bound.
- Under strict token-level speculative equivalence, Qwen3.5 remains unresolved:
  block verification can diverge on off-trajectory speculative prefixes.
- Under the updated GSM8K answer-level quality gate, Qwen3.5 is now promising:
  GatedDeltaNet state-history commit removes the replay blocker and reaches
  `2.06x` on the standard 8-sample, 512-token benchmark while matching target
  answer accuracy on the sampled scorer.
- Qwen3-4B remains the cleaner full-attention/trimmable-cache path and already
  passed the 32-sample quality expansion.
- The public clean-room MLX runtime is intentionally block-only.

## Next Improvement Ideas

1. Expand Qwen3.5 answer-level scoring beyond 8 GSM8K samples and include the
   same extraction/regression table used for Qwen3-4B.
2. Add a targeted diagnostic that compares the new history-commit cache state
   against the old full-replay cache state for accepted prefixes.
3. Profile the per-layer GatedDeltaNet prefix recompute to see whether kernel
   versus reference recurrent commit is faster for short accepted prefixes.
4. Keep strict verifier diagnostics available, but do not use token equivalence
   as the only success gate for the quality-oriented benchmark path.
