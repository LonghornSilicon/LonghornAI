# Sampling (`src/kernels/sampling.{hpp,cpp}`)

## What it computes
- `splitmix64` PRNG (deterministic, platform-uniform).
- In-place logit filters: `apply_temperature`, `apply_top_k`,
  `apply_top_p`, `apply_min_p`, `apply_typical_p`.
- `argmax_sample` (greedy), `softmax_sample` (multinomial),
  `sample` (full pipeline).

## Reference path
Each filter has a closed-form set-membership test verified by hand-
constructed tests.

## Memory traffic
Operates on a single `vocab`-length logit row (~30k–256k floats).
- Top-k: one nth_element pass.
- Top-p: one sort + one cumulative pass.
- Min-p / typical-p: one or two passes through the row.

## Arithmetic intensity
Trivial — one to a few FLOPs per element. **Memory-bound for vocab
> ~1k.**

## Roofline class
**Memory-bound.** Sampling is per-token at decode and the vocab is
massive; bandwidth to read the logit row dominates.

## Engine assignment
**Reduction Engine + Vector Unit.** The argmax / top-k / sort use the
reduction primitives; the filter masks are vector pointwise. A small
dedicated **Sampling Engine** that fuses temperature → top-k →
top-p → softmax → multinomial would save bandwidth by streaming the
logits exactly once per token.

## SRAM working set
One logit row in scratch (`vocab · 4` bytes). For vocab=128k that's
512 KiB — too large for L1 but fits in L2. The fused-sample engine
streams it.

## DMA pattern
**Sequential streaming** of the logit row out of the LM head's GEMM
output.

## Datapath dtype requirements
- Logits: FP32 (precision matters at the tail of the distribution).
- Output: int32 (token id).

## Open silicon questions
- Whether to fuse the LM-head GEMM's epilogue with the sampler.
  Eliminates the logit-row round-trip to DRAM — large bandwidth win
  for decode.
