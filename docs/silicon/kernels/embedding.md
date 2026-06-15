# Embedding (`src/kernels/embedding.{hpp,cpp}`)

## What it computes
`out[t, :] = scale · table[ids[t], :]`. Out-of-range ids produce a
zero row.

## Reference path
`embedding`.

## Memory traffic
- Reads: `n_ids · dim` (scattered into the table).
- Writes: `n_ids · dim` (contiguous output).
- The table itself is **vocab × dim** in DRAM; effective working set
  per call is the touched rows.

## Arithmetic intensity
~1 FLOP/elem (scale). **Bandwidth-limited gather.**

## Roofline class
**Memory-bound** (latency-limited at decode batch=1; bandwidth-limited
at prefill batch).

## Engine assignment
**DMA Engine.** This is row-gather work, not arithmetic. Cleanly
offloaded to a scatter-gather DMA with FP32 → BF16/FP16 downcast in
the read path if the table is mixed-precision.

## SRAM working set
The set of touched rows. For a prefill of 4k tokens with vocab 32k,
a typical hit pattern is ~3–4k unique ids; that's ~12–16 MiB at FP16
hidden=4096 — **does not fit in L1**. Lives in HBM; gather
streams it.

## DMA pattern
**Indirect / scatter-gather** with `(id × dim, dst)` descriptors. A
strided alternative is to materialize a token-id permutation and run a
strided copy.

## Datapath dtype requirements
- Table: BF16 / FP16 / FP8 (any).
- Output: same as table (or upcast to the next layer's compute dtype).

## Open silicon questions
- Whether to dedicate a small embedding cache for hot tokens
  (BOS/EOS/system tokens) — see Phase 6 §6.2 prompt caching.
