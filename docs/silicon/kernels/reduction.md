# Reduction (`src/kernels/reduction.{hpp,cpp}`)

## What it computes
Last-axis reductions: `reduce_sum`, `reduce_max`, `reduce_mean`,
`argmax`, `topk` (descending, lowest-index tie-break).

## Reference path
- `reduce_sum/max/mean` — single-pass.
- `argmax` — single-pass with strict-greater compare for tie-break.
- `topk` — length-k min-heap, then sort survivors descending.

## Memory traffic
- Reads: `rows · dim` floats.
- Writes: `rows` (sum/max/mean), `rows` (argmax), `rows · k` (topk).

## Arithmetic intensity
~1 FLOP/elem on the input; the heap update is `O(log k)` per element
in `topk`. **Memory-bound** in all variants.

## Roofline class
**Memory-bound.** Top-k for sampling (`k ≪ vocab`) is a near-zero-FLOP
streaming kernel.

## Engine assignment
**Reduction Engine.** Tree-reduce hardware (sum / max / mean) plus a
small comparator network for top-k.

## SRAM working set
- Sum/Max/Mean: one accumulator per row (constant).
- Top-k: a length-k heap per row. For sampling-kit top-50 over a row of
  vocab=32k: heap size 50 × 8 bytes = 400 bytes per active row. Trivial.

## DMA pattern
**Sequential streaming.**

## Datapath dtype requirements
- Input: any.
- **Reduction lanes: FP32** (or INT32 for argmax).

## Open silicon questions
- Whether to fuse `argmax` into the softmax engine's epilogue (greedy
  sampling requires `softmax → argmax`; if greedy is the only sampler,
  the softmax can be skipped entirely).
