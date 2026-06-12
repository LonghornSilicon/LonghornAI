# Activation Family — Kernel Contract

> PLAN.md §3 Phase 1 — foundational compute kernels.
> Operators: `gelu`, `silu`.

## Operators

### `gelu(x, approximate="none")`
Gaussian-error-linear-unit activation, applied elementwise.

* `approximate="none"` — exact form, `0.5 * x * (1 + erf(x / √2))`.
* `approximate="tanh"` — tanh form, `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))`.

Both variants ship with their own golden (`gelu` / `gelu_tanh`).

### `silu(x)`
SiLU / swish: `x * sigmoid(x)`. The activation in SwiGLU MLPs (Llama, Mistral,
Qwen, DeepSeek).

## Shape contract

Both activations are shape-preserving and dtype-preserving. Output shape and
dtype match the input.

## Supported dtypes

| dtype     | Inputs | Output |
|-----------|:------:|:------:|
| float64   |   ✓    | float64 |
| float32   |   ✓    | float32 |
| float16   |   ✓    | float16 |
| bfloat16  |   ✓    | bfloat16 (emulated) |

## Numerical contract

Validated against `kernels/activation/reference.py`. Note that `gelu(approximate="tanh")` is **not** an approximation of the exact GELU under our tolerance — it is a separate kernel with its own golden, since it has framework-PyTorch parity as the reference (Llama/Gemma use tanh-GELU; the contract is bit-compat with that).

## Performance contract

Memory-bound elementwise — roofline target ≥ 85% of bandwidth roof on the
production target. Per-element cost is one read + one write at the I/O dtype
plus the math (≈ 8 FLOPs/elem for GELU/SiLU).

These activations are prime candidates for **fusion** with their producing
GEMM (e.g. `up_proj → SiLU → gate · up`) — the fused-kernel framework arrives
in Phase 6 (PLAN.md §3 Phase 6).

## Tuning

`kernels/activation/tuning.py` defines `(vec_width, approximate)` for `gelu`
and `(vec_width)` for `silu`.
