# Quantization utilities (`src/kernels/quant.{hpp,cpp}`)

## What it computes
- INT8 symmetric: per-tensor / per-row / per-column quant + dequant.
- INT4 group: packed two-per-byte, group-along-K with per-`(group, col)`
  fp32 scales.
- FP8 E4M3: per-tensor / per-row / per-column quant + dequant.
- `q4_get` inline unpack helper.

## Reference path
Direct, single-pass; no separate optimised paths.

## Memory traffic
Two-pass for per-row / per-col / per-group scopes (one to find max-abs,
one to scale + write). Per-tensor is one pass + a max-abs reduction.

## Arithmetic intensity
~2 FLOPs/elem (one mul to scale + one abs/compare). **Memory-bound.**

## Roofline class
**Memory-bound.** Quantization is plumbing for the actual compute
kernels; on silicon it lives inline with the GEMM input feeders.

## Engine assignment
**Vector Unit + GEMM input-feeder.** On the inference path, weights
arrive pre-quantized (offline calibration produced AWQ / GPTQ / GGUF
artifacts); only activation quantization runs at inference time, and
that fuses into the previous layer's epilogue.

## SRAM working set
Per-row: a row buffer for the scan-then-scale pass. Per-group: a
group buffer (typically 32 / 64 / 128 elements).

## DMA pattern
**Sequential streaming.** Output writes are tighter than input reads
when packing INT4 (2× compression).

## Datapath dtype requirements
- Inputs: FP32 (the source of truth for calibration).
- Scales: **FP32 hard requirement**.
- Output: INT8 / INT4 (packed) / FP8 (per dtype).

## Open silicon questions
- Whether to expose calibration-time quantization on-chip at all, or
  treat all quantization as offline (host-side) and never run these
  kernels at inference time.
