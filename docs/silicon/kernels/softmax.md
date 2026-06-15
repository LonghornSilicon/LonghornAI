# Softmax (`src/kernels/softmax.{hpp,cpp}`)

## What it computes
`y_i = exp(x_i − max(x)) / sum_j exp(x_j − max(x))` per row.

## Reference path
- `softmax_ref` (textbook two-pass).
- `softmax` (single fused pass).

## Memory traffic
- Reads: `rows · dim` floats.
- Writes: `rows · dim` floats.

## Arithmetic intensity
~5 FLOPs/element (max compare, subtract, exp, sum, divide). **AI ≈ 0.6
FLOPs/byte.**

## Roofline class
**Memory-bound** for short rows. Long rows (vocab-scale, ~32k) are
exp-bound on CPU because the polynomial-approx exp is slower than the
load; on silicon the streaming exp pipeline absorbs this.

## Engine assignment
**Softmax Engine.** Online streaming form (`max-subtract → exp → sum →
reciprocal`) is the canonical fused pipeline. The same engine consumes
attention scores in `flash_attention` and `paged_attention`.

## SRAM working set
One row buffer `dim · 4` bytes. For attention scores the tile size is
chosen to fit `block_k` columns alongside the (m, ℓ) state — see
`attention.md`.

## DMA pattern
**Sequential streaming** by row.

## Datapath dtype requirements
- Input: FP32 logits (post-attention scale or post-softmax pre-sample).
- Reduction lanes: **FP32 hard requirement** (subtract-max preserves
  ulps; sum needs the headroom).

## Open silicon questions
- Polynomial-approx `exp` precision vs. table-lookup. The former wins on
  area; the latter on uniformity of latency.
