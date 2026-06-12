# GEMM Family — Kernel Contract

> PLAN.md §3 Phase 1 — foundational compute kernels.
> Operators: `gemm`, `batched_gemm`, `grouped_gemm`, `tensor_contraction`.

## Operators

### `gemm(a, b, c=None, alpha=1.0, beta=0.0)`
`C := alpha * (A @ B) + beta * C`.

| Operand | Shape   | Layout    | Notes |
|---------|---------|-----------|-------|
| A       | (M, K)  | row-major | required |
| B       | (K, N)  | row-major | required |
| C       | (M, N)  | row-major | optional bias / accumulator |
| Output  | (M, N)  | row-major | dtype = `np.result_type(a, b)` |

### `batched_gemm(a, b, alpha=1.0)`
Uniform-shape batched matmul. `A: (..., M, K)`, `B: (..., K, N)`. Leading batch
dims must broadcast; output is `(..., M, N)`.

### `grouped_gemm(a_list, b_list, alpha=1.0)`
Variable-shape grouped matmul — the MoE expert-matmul primitive. `len(a_list)
== len(b_list)`; each pair has matching inner dim. Returns a list of
`(M_i, N_i)` outputs.

### `tensor_contraction(a, b, subscripts)`
Generalized einsum-style contraction over the GEMM core. `subscripts` follows
the `np.einsum` grammar.

## Supported dtypes

| dtype     | Inputs | Accumulation (CPU ref) | Output |
|-----------|:------:|:----------------------:|:------:|
| float64   |   ✓    | float64                | float64 |
| float32   |   ✓    | float64*               | float32 |
| float16   |   ✓    | float64*               | float16 |
| bfloat16  |   ✓    | float64*               | bfloat16 (emulated; see `validation/bf16.py`) |

\* The CPU reference backend accumulates in float64 — it is the correctness
anchor (PLAN.md §2.2). Hardware-representative backends (sim/lhsil) use FP32
accumulation per PLAN.md §3 Phase 1's "FP16/BF16/FP8 in, FP32 accum" contract.

## Numerical contract

Validated against `kernels/gemm/reference.py` under the per-dtype tolerance
policy in `validation/tolerance.py`:

| dtype     | rtol  | atol  |
|-----------|------:|------:|
| float64   | 1e-12 | 1e-12 |
| float32   | 1e-5  | 1e-6  |
| float16   | 1e-2  | 1e-2  |
| bfloat16  | 2e-2  | 2e-2  |

Failure on the differential test is a contract violation, not a tolerance
question.

## Performance contract

Per PLAN.md §9.1 (silicon target):

| Metric                                | Target |
|---------------------------------------|-------:|
| GEMM % of cuBLAS-class peak (FP16/BF16) | ≥ 90%  |
| GEMM % of cuBLAS-class peak (FP8/INT8)  | ≥ 85%  |
| Roofline efficiency (compute-bound)     | ≥ 80%  |
| Roofline efficiency (memory-bound)      | ≥ 85%  |

The shape suite that evaluates these targets lives in
`benchmarks/shape_suite.py` and is run with `lhai bench-suite`. The recorded
baseline is in `benchmarks/gemm_baseline.json`. M1 captures the CPU NumPy
reference baseline; production targets land on silicon.

## Tuning

`kernels/gemm/tuning.py` defines `GEMM_TUNING_SPACE` over
`(tile_m, tile_n, tile_k, stages)`. The CPU backend treats these as ABI
placeholders; sim/RTL/FPGA/lhsil backends consume them through dispatch.

`autotune_gemm(M, K, N, dtype)` runs grid search and returns the best config
— the Phase-1 "tuning framework operational for GEMM" exit-gate item.
