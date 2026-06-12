# Normalization Family — Kernel Contract

> PLAN.md §3 Phase 1 — foundational compute kernels.
> Operators: `layernorm`, `rmsnorm`.

## Operators

### `layernorm(x, weight=None, bias=None, eps=1e-5, axis=-1)`
Layer normalization over `axis` with optional affine `weight` * x + `bias`.

| Operand | Shape           | Notes |
|---------|-----------------|-------|
| x       | (..., D)        | reduction over `axis` |
| weight  | shape of axis   | optional gain |
| bias    | shape of axis   | optional offset |
| Output  | shape of x      | dtype = x.dtype |

### `rmsnorm(x, weight=None, eps=1e-6, axis=-1)`
Root-mean-square normalization (Llama / Qwen / Mistral norm). Same shape
contract as `layernorm`; no bias term.

## Reductions

* Mean and variance computed in float64 on the CPU reference (Welford-stable
  semantics; PLAN.md §3 Phase 1).
* Hardware-representative backends compute reductions in FP32 with a
  Welford-pair tree.

## Supported dtypes

| dtype     | Inputs | Reduction (CPU ref) | Output |
|-----------|:------:|:-------------------:|:------:|
| float64   |   ✓    | float64             | float64 |
| float32   |   ✓    | float64             | float32 |
| float16   |   ✓    | float64             | float16 |
| bfloat16  |   ✓    | float64             | bfloat16 (emulated) |

## Numerical contract

Validated against `kernels/normalization/reference.py` under the per-dtype
tolerance policy. `eps` is part of the numerical contract: changing it changes
the golden output.

## Performance contract

Both kernels are **memory-bound**. The roofline target is ≥ 85% of the
bandwidth roof on the production target (PLAN.md §9.1). Per-token cost is
`2 * D` reads + `D` writes (plus `D` weight/bias reads when affine is fused).

Fusion with subsequent projections (e.g. `RMSNorm + QKV proj`) is the
single highest-leverage decode optimization — landed in Phase 6 fused
kernels (PLAN.md §3 Phase 6, §6.1).

## Tuning

`kernels/normalization/tuning.py` defines spaces over `(block_size,
vec_width, fuse_affine)`. CPU backend ignores them; lhsil consumes them
through dispatch.
