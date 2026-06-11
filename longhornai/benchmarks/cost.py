"""Per-kernel cost models: FLOPs and bytes moved.

These analytic models feed the roofline classification in the harness — they
determine whether a kernel is compute- or memory-bound and how far it sits from
the relevant roof (PLAN.md §4.2, §6.2). Bytes assume each distinct operand is
read once and each output written once at the given dtype width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Analytic cost of one kernel invocation."""

    flops: float
    bytes_moved: float

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte — the x-axis of the roofline."""
        return self.flops / self.bytes_moved if self.bytes_moved else float("inf")


def _itemsize(dtype) -> int:
    return np.dtype(dtype).itemsize


def cost_gemm(m: int, k: int, n: int, dtype=np.float16) -> CostModel:
    """GEMM C[M,N] = A[M,K] @ B[K,N]: 2*M*N*K FLOPs."""
    isz = _itemsize(dtype)
    flops = 2.0 * m * n * k
    bytes_moved = isz * (m * k + k * n + m * n)
    return CostModel(flops=flops, bytes_moved=bytes_moved)


def cost_elementwise(num_elems: int, flops_per_elem: float, dtype=np.float16) -> CostModel:
    """Memory-bound elementwise op: read + write each element."""
    isz = _itemsize(dtype)
    return CostModel(flops=flops_per_elem * num_elems, bytes_moved=2.0 * isz * num_elems)


def cost_softmax(num_elems: int, dtype=np.float16) -> CostModel:
    # ~5 flops/elem (max, sub, exp, sum, div); read + write.
    return cost_elementwise(num_elems, flops_per_elem=5.0, dtype=dtype)


def cost_for(op: str, dtype=np.float16, **dims) -> CostModel:
    """Convenience dispatcher used by the CLI for common ops."""
    if op == "gemm":
        return cost_gemm(dims["m"], dims["k"], dims["n"], dtype=dtype)
    if op == "softmax":
        return cost_softmax(dims["num_elems"], dtype=dtype)
    if op in ("layernorm", "rmsnorm", "gelu", "silu", "rope"):
        return cost_elementwise(dims["num_elems"], flops_per_elem=8.0, dtype=dtype)
    raise KeyError(f"no cost model for op '{op}'")
