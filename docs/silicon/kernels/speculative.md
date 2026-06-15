# Speculative decoding (`src/kernels/speculative.{hpp,cpp}`)

## What it computes
- `speculative_verify` — canonical Leviathan/Chen 2023 accept/reject:
  per draft token compute `a = min(1, p_target / p_draft)`, accept iff
  `uniform() < a`; on reject, sample bonus from `max(0, p_target -
  p_draft)` renormalised; if all accepted, sample fresh from
  `p_target[K]`.
- `build_tree_attention_bias` — per-node ancestor-reachability mask
  over a candidate-token tree (Medusa, Eagle, generic tree decoding).

## Reference path
Both validated with statistical / set-membership tests; no separate
naive reference for verify (the algorithm itself is small enough to
audit).

## Memory traffic
- Verify: read `K · vocab + (K+1) · vocab` probability rows; write a
  single token id. Per-step `vocab` work is dominant.
- Tree mask: `n_nodes · (n_history + n_nodes) · 4` bytes written once
  per verification batch. Small.

## Arithmetic intensity
- Verify: O(K · vocab) reads, O(K) decisions, plus one resample over
  vocab. **Memory-bound.**
- Tree mask: a constant write per (q, k) cell. **Memory-bound.**

## Roofline class
**Memory-bound** for both. Both pieces are small relative to the
target-model forward pass they enable; the speedup comes from running
the target forward over K+1 positions in *one pass* using the tree
mask.

## Engine assignment
- Verify: **Vector Unit** (per-vocab scan) + **PRNG** + **Reduction**.
  A small dedicated **Speculative Sampler** could fuse verify with the
  upstream LM head GEMM (similar to the sampling fusion).
- Tree mask: **Host-side / Scheduler.** Constructed once per
  verification batch and uploaded as the AttnConfig::bias buffer.

## SRAM working set
- Verify: two `vocab`-length probability rows for the active position.
  Compact.
- Tree mask: `n_nodes^2 + n_nodes · n_history` floats. For n_nodes=64
  (Medusa) and n_history=2048: ~520 KiB. Sits in shared L2.

## DMA pattern
- Verify: **streamed** from the target/draft probability buffers.
- Tree mask: **broadcast** to all attention heads.

## Datapath dtype requirements
- Probabilities: **FP32** (rejection ratio precision matters at the
  tail).
- Mask: FP32 (additive into attention scores).

## Open silicon questions
- Whether to expose tree-mask construction on-chip at all. The host
  can build it; the only argument for on-chip construction is when
  speculative trees are dynamic per step (Eagle's feature-conditioned
  trees).
