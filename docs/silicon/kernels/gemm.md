# GEMM (`src/kernels/gemm.{hpp,cpp}`)

## What it computes
`C[M, N] = α · A[M, K] · B[K, N] + β · C[M, N]`. Plus uniform-shape
batched and ragged grouped variants.

## Reference path
- `gemm_ref` (triple loop) — golden numerics.
- `gemm` (4-level blocked) — production path.
- `gemm_batched`, `gemm_grouped` — batched/MoE shapes.

## Memory traffic
- Reads: `M·K + K·N` (A and B), `M·N` if `β ≠ 0`.
- Writes: `M·N`.
- Reuse: each A row reused N times, each B col reused M times under good
  blocking. Ridge of the kernel.

## Arithmetic intensity
`AI = 2·M·N·K / (4·(M·K + K·N + M·N))`. For a 128×768×768 tile,
AI ≈ 48 FLOPs/byte. **Compute-bound.**

## Roofline class
**Compute-bound** for square-ish shapes. Skinny GEMV (M=1) drops the
reuse term and falls into memory-bound territory — see `gemv` below
when added.

## Engine assignment
**Tensor Unit.** A systolic array with FP16/BF16/FP8/INT8/INT4 inputs
and FP32/INT32 accumulators is the right shape; A streams down,
B is held / pre-loaded as the panel.

## SRAM working set
A panel (`MR × KR` of A) + a panel (`KR × NR` of B) + an
accumulator tile (`MR × NR` in FP32). For `MR=NR=64, KR=64`:
`64·64·2 (A) + 64·64·2 (B) + 64·64·4 (C) = 32 KiB`. Sits comfortably
in L1/L0.

## DMA pattern
**Strided streaming** of A row tiles + B column-panel multicast. No
gather/scatter needed.

## Datapath dtype requirements
- Inputs: any of FP16/BF16/FP8/INT8/INT4 (per the quantized variants).
- Multipliers: input-width.
- **Accumulator: FP32 (or INT32 for W8A8).** Hard requirement.

## Open silicon questions
- Right `MR × NR × KR` micro-tile for the target SRAM budget.
- Whether to share the systolic array between dense GEMM and grouped
  GEMM (MoE) or have separate arrays for fine-grained MoE.
