"""Phase-1 foundational compute kernels (PLAN.md §3 Phase 1).

These are the stable public kernel entry points. Each is a thin front-end that
dispatches through :func:`longhornai.runtime.dispatch` to the active backend, so
the same call re-targets across CPU / sim / RTL / FPGA / silicon.

Every kernel has a matching float64 golden in :mod:`longhornai.validation` and a
cost model in :mod:`longhornai.kernels.cost` used by the benchmark harness.
"""

from __future__ import annotations

from .gemm import batched_gemm, gemm, grouped_gemm, tensor_contraction
from .normalization import layernorm, rmsnorm
from .activation import gelu, silu
from .elementwise import embedding_lookup, reduce, rope, softmax

__all__ = [
    "gemm",
    "batched_gemm",
    "grouped_gemm",
    "tensor_contraction",
    "layernorm",
    "rmsnorm",
    "gelu",
    "silu",
    "softmax",
    "rope",
    "embedding_lookup",
    "reduce",
]
