# Quantized GEMM (`src/kernels/gemm_quant.{hpp,cpp}`)

## What it computes
- `gemm_w8a8` — INT8 weights × INT8 activations, INT32 accumulator,
  fp32 rescale at the epilogue.
- `gemm_w4a16_groupwise` — INT4-group weights × FP32 activations, with
  on-the-fly dequant per K-group panel.
- `gemm_fp8_e4m3` — FP8 (E4M3) × FP8 (E4M3), FP32 accumulator,
  per-tensor scale combined at the epilogue.

All produce FP32 output. **Accumulator precision is sacred.**

## Reference path
Each matches FP32 GEMM within a documented relative-RMS tolerance:
W8A8 < 5%, W4A16 < 10%, FP8 < 10% on uniform-random unit-variance data.

## Memory traffic
- W8A8: `M·K + K·N` bytes inputs (1 byte/elem) + per-row & per-col
  scales (negligible). Halves the weight read vs FP16.
- W4A16: `M·K · 4` (FP32 acts) + `K·N / 2` (packed weights) +
  `(K/G) · N · 4` scales. **>3× weight bandwidth reduction vs FP16.**
- FP8: `M·K + K·N` bytes (1 byte/elem). Halves weights vs FP16.

## Arithmetic intensity
Same FLOPs as FP32 GEMM but with reduced byte traffic on the input
side, so AI **rises** by the byte ratio (~2× for W8A8/FP8, ~4× for
W4A16). Pushes memory-bound shapes (small M / decode GEMVs) closer to
compute-bound.

## Roofline class
- Compute-bound at large shapes.
- Decode GEMV with W4A16 weights: still memory-bound but **dramatically
  less so**. The exact reason quantization is the highest-leverage
  decode-bandwidth win.

## Engine assignment
**Tensor Unit** (with dequant on the input feeders) + **Rescale Unit**
in the epilogue. The MAC array sees a uniform compute dtype regardless
of storage; the input feeders unpack INT4 / INT8 / FP8 on the fly.

## SRAM working set
Same panel sizes as FP32 GEMM (~32 KiB), but the B-panel is half (FP8 /
INT8) or quarter (INT4) the size. More B reuse per byte transferred.

## DMA pattern
**Strided streaming.** INT4 packed reads are unaligned at the byte
level inside the unpack stage; the DMA delivers raw bytes and the
unpack happens on the input feeder.

## Datapath dtype requirements
- W8A8: int8 multipliers, **INT32 accumulator**, FP32 rescale.
- W4A16: int4 unpack → BF16 dequant → FP16 multiplier → **FP32
  accumulator**.
- FP8: FP8 multipliers, **FP32 accumulator**.

## Open silicon questions
- Whether to support W4A8 (compose W4A16 dequant with the W8A8 path)
  or treat it as future work.
- Group-size policy for INT4: 32 / 64 / 128. Smaller groups improve
  fidelity at the cost of more scale memory.
