# Sparse Attention Family — Kernel Contract

> PLAN.md §3 Phase 6 / §8 M7 — sparse attention.
> Operators: `sliding_window_attention`, `block_sparse_attention`,
> `dilated_attention`.

## Why

Dense attention is O(S²); long-context decoding is dominated by KV reads.
Sparse patterns drop attention to **O(S × window)** or **O(S × density)**
with negligible quality loss for the right architecture:

* **Sliding window** — Mistral, Gemma-2 (alternating local/global). Each
  query attends only to the last `window` keys.
* **Block sparse** — Longformer / BigBird. The user supplies a
  per-(Q-block, K-block) mask; non-masked blocks compute SDPA, masked
  blocks contribute `-inf`.
* **Dilated** — Sparse Transformer family. Strided causal attention.

## Operators

### `sliding_window_attention(q, k, v, *, window, scale=None)`
Causal sliding-window: query `i` attends to keys in `[max(0, i - window
+ 1), i]`. Same `(..., S, D)` operand shapes as `sdpa`. `window` is the
context size — Mistral-7B uses 4096.

### `block_sparse_attention(q, k, v, *, block_mask, block_size, scale=None, causal=False)`
Per-(Q-block, K-block) boolean mask:

| Operand    | Shape                             | dtype |
|------------|-----------------------------------|-------|
| block_mask | (S_q // block_size, S_kv // block_size) | bool  |

`True` means *attend* to that block; `False` means *masked out*. `S_q`
and `S_kv` must be divisible by `block_size`. `causal=True` AND-masks the
above-diagonal positions on top of the block mask.

### `dilated_attention(q, k, v, *, dilation, scale=None)`
Causal + stride-`dilation`: query `i` attends to keys at
`{i, i - d, i - 2d, ...}`.

## Numerical contract

Each variant is SDPA-with-mask in float64; validated against the float64
reference. Test inputs must keep at least one un-masked column per Q row
(see attention KERNEL.md — entirely-masked rows produce `NaN` in math
form and are undefined input).

## Performance contract

Sparse attention's win is **bandwidth + compute** proportional to mask
density:

* Sliding window: `O(S × window)` — typically ~10× faster on long context.
* Block sparse: `O(S × S × density)` — depends on mask.
* Dilated: `O(S² / d)`.

The lhsil implementation skips masked tiles entirely; the CPU reference
computes the full score matrix and applies the mask (correctness path).

## Cross-target equivalence

PLAN.md §8 M7 exit gate: silicon bring-up must pass cross-target
equivalence on every kernel, including sparse attention. The harness in
`validation/equivalence.py` runs each variant under all 5 backends.
