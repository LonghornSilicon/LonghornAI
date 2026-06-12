# Elementwise / Sequence Family — Kernel Contract

> PLAN.md §3 Phase 1 — foundational compute kernels.
> Operators: `softmax`, `rope`, `embedding_lookup`, `reduce`.

## Operators

### `softmax(x, axis=-1)`
Numerically stable softmax (online / streaming-stable form: subtract max
before `exp`). The numerical kernel attention depends on (PLAN.md §3 Phase 2);
correctness here is load-bearing for FlashAttention.

### `rope(x, positions=None, theta=10000.0)`
Rotary position embedding on `x` of shape `(..., seq, head_dim)`. `head_dim`
must be even. `positions` defaults to `arange(seq)`. The current
implementation uses the **half-rotation** layout (split last dim in halves);
the **interleaved** layout is part of the tuning space.

### `embedding_lookup(table, ids, scale=1.0)`
Gather rows of `table` indexed by integer `ids`, optionally scaled. Output
preserves `table.dtype`; ids may be any integer dtype.

### `reduce(x, op, axis=-1, keepdims=False)`
Reduction primitive backing norms and softmax. `op` ∈ {"sum", "max", "mean"}.

## Supported dtypes

| dtype     | softmax | rope | embedding | reduce |
|-----------|:-------:|:----:|:---------:|:------:|
| float64   |   ✓     |  ✓   |    ✓      |   ✓    |
| float32   |   ✓     |  ✓   |    ✓      |   ✓    |
| float16   |   ✓     |  ✓   |    ✓      |   ✓    |
| bfloat16  |   ✓     |  ✓   |    ✓      |   ✓    |

`reduce` and `softmax` accumulate in float64 on the CPU reference.

## Numerical contract

Validated against `kernels/elementwise/reference.py`. RoPE has a structural
invariant in addition to its golden: it is a **rotation**, so per-position
output norms must match input norms (verified by `tests/test_kernels.py`).

## Performance contract

All four are memory-bound — roofline target ≥ 85% of bandwidth roof on the
production target (PLAN.md §9.1).

* **Softmax** is the inner kernel of attention; it must fuse with the
  attention score path in FlashAttention v1/v2 (PLAN.md §3 Phase 2).
* **RoPE** typically fuses with the QKV projection (Phase 6 fusions).
* **Embedding lookup** scales with vocab size; large-vocab models (Llama-3,
  Gemma) demand block-table-friendly access patterns at the lhsil backend.
* **Reduce** is the building block; it must vectorize cleanly on hardware.

## Tuning

Spaces in `kernels/elementwise/tuning.py` cover `block_size`, `vec_width`,
softmax `online` mode, and RoPE `layout` (interleaved vs half).
