# Tensor transforms (`src/kernels/transform.{hpp,cpp}`)

## What it computes
`transpose2d`, `permute` (N-D), `concat`, `split`, `gather_rows`,
`scatter_add_rows`. Layout-only operations (transpose/permute/concat/
split) plus index ops (gather/scatter).

## Reference path
Direct, single-pass each.

## Memory traffic
- Transpose / permute / concat / split: read N + write N (no compute).
- Gather: indirect read N + contiguous write N.
- Scatter-add: read N + read-modify-write target rows (with potential
  dependency on duplicate indices).

## Arithmetic intensity
**Effectively zero.** Pure data motion.

## Roofline class
**Bandwidth-limited.** Latency-limited at small sizes.

## Engine assignment
**DMA Engine.** Strided / scatter-gather / transposed-tile modes
eliminate the need to execute these on the compute fabric. This is one
of the highest-leverage silicon decisions: a strong DMA absorbs all of
this.

## SRAM working set
- Transpose: a `MR × NR` tile in scratch (e.g. 32 × 32 FP32 = 4 KiB)
  to avoid cache thrash on the read-side strided pattern.
- Permute: same idea with the tile aligned on the contiguous trailing
  dim.
- Concat / split: streaming; no scratch.

## DMA pattern
- Transpose / permute: **transposed-tile DMA** (read N rows, write
  M rows; tile-internal transpose cheap on a small register file).
- Concat / split: **strided streaming**.
- Gather: **scatter-gather** with a descriptor list.
- Scatter-add: **atomic scatter** or two-phase (sort by destination,
  then run a sequential add).

## Datapath dtype requirements
- Element width: any. The DMA preserves bits.

## Open silicon questions
- Whether scatter-add's atomic path is worth the silicon cost vs. the
  software-side sort-then-reduce alternative (relevant for MoE combine
  with duplicate-token routing).
