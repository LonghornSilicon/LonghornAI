# Quantized GEMM Family — Kernel Contract

> PLAN.md §3 Phase 4 / M5 — quantization kernels.
> Operators: `gemm_w8a8`, `gemm_w4a16`, `activation_quantize`,
> `activation_dequantize`.

## Operators

### `gemm_w8a8(a_q, b_q, *, scale_a, scale_b, zero_point_a=None, zero_point_b=None, out_dtype=fp16)`
INT8 weights × INT8 activations, INT32 accumulate, FP scale to output dtype.

| Operand     | Shape      | dtype | Notes |
|-------------|------------|-------|-------|
| a_q         | (M, K)     | int8  | activations (dynamic-quantized) |
| b_q         | (K, N)     | int8  | weights (calibrated; SmoothQuant-friendly) |
| scale_a     | () or (M,1)| fp32  | per-tensor or per-row activation scale |
| scale_b     | () or (1,N)| fp32  | per-tensor or per-column weight scale |
| zero_point_a / _b | matching shape | int32 | optional asymmetric |
| Output      | (M, N)     | fp16/fp32 | dtype = `out_dtype` |

### `gemm_w4a16(a, b_q_packed, *, scale_b, zero_point_b=None, group_size, K, out_dtype=None)`
INT4 packed weights × FP activations. The dominant LLM-decode quant kernel:
weight bandwidth shrinks ~4× and the dequant fuses into the matmul prologue
on real hardware.

| Operand        | Shape           | dtype | Notes |
|----------------|-----------------|-------|-------|
| a              | (M, K)          | fp16/fp32 | activations stay FP |
| b_q_packed     | (K // 2, N)     | int8  | two INT4 nibbles per byte |
| scale_b        | (K // group_size, N) | fp32 | groupwise weight scales |
| zero_point_b   | matching shape  | int32 | optional asymmetric |
| Output         | (M, N)          | fp16/fp32 | dtype defaults to `a.dtype` |

`group_size` is part of the tuning space — typically 32, 64, or 128.

### `activation_quantize(x, *, bits=8, axis=-1, asymmetric=False)`
Returns `(q, scale[, zero_point])`. Dynamic per-row symmetric or
asymmetric INT quant; the prologue of the W8A8 path.

### `activation_dequantize(q, *, scale, zero_point=None, dtype=fp16)`
Inverse of `activation_quantize` — fused into the W8A8 epilogue on real
hardware.

## Numerical contract

| Path     | Tolerance vs FP32 reference | Notes |
|----------|----------------------------:|-------|
| W8A8     | ≤ 1% perplexity delta vs FP16 | bounded by SmoothQuant calibration |
| W4A16    | ≤ defined per-model bound     | bounded by AWQ / GPTQ calibration |

The differential harness validates the *kernel math* against the float64
reference; calibration accuracy is validated end-to-end in
:mod:`longhornai.quantization.calibration`.

## Performance contract

PLAN.md §9.1:

| Metric                                           | Target |
|--------------------------------------------------|-------:|
| GEMM % of cuBLAS-class peak (FP8/INT8)           | ≥ 85% |
| INT4/INT8 accuracy at ≥ measured target speedup  | M5 exit gate |

Decode is bandwidth-bound; W4A16 is the principal bandwidth lever
(4× weight read reduction).

## Tuning

`kernels/quant/tuning.py`:

* W8A8: `(tile_m, tile_n, tile_k, split_k)`.
* W4A16: `(tile_m, tile_n, group_size, fuse_dequant)`.
* activation_quantize: `(vec_width, fuse_into_prologue)`.
