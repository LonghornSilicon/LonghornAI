# Normalization (`src/kernels/normalization.{hpp,cpp}`)

## What it computes
- LayerNorm: `y = (x − μ) · rsqrt(σ² + ε) · γ + β`.
- RMSNorm: `y = x · rsqrt(mean(x²) + ε) · γ`.

Both row-wise; reductions accumulate in FP32.

## Reference path
- `layernorm_ref` (Welford one-pass mean/variance), `layernorm`.
- `rmsnorm_ref`, `rmsnorm`.

## Memory traffic
- Reads: `rows · dim` (x) + `dim` (γ, β).
- Writes: `rows · dim` (y).
- Per row: 2 passes over `dim` floats + a few scalars per row.

## Arithmetic intensity
~3 FLOPs/element + amortised `rsqrt`. **AI ≈ 0.4 FLOPs/byte** (measured
via `bench_main`'s rmsnorm row).

## Roofline class
**Memory-bound.** Sustained on a developer x86 core at ~15 GB/s, ~70%
of single-thread DRAM bandwidth.

## Engine assignment
**Normalization Engine.** A streaming reduction lane: Welford accumulator
→ rsqrt → broadcast scale + bias. Welford eliminates the second pass that
the textbook variance formula needs.

## SRAM working set
One row buffer (`dim · 4` bytes for x, plus `dim · 4` for γ, β). For
`dim = 4096` that's 48 KiB total; comfortably in L1.

## DMA pattern
**Sequential streaming** of x rows; γ, β preloaded once and held for the
whole batch.

## Datapath dtype requirements
- Input: BF16/FP16/FP32.
- Reduction lanes: **FP32 hard requirement** (Welford is unstable
  otherwise).
- Output: input dtype (downcast at write).

## Open silicon questions
- Whether to fuse RMSNorm + RoPE (next op in Llama / Mistral) into the
  same engine pass to halve memory traffic on the residual stream.
