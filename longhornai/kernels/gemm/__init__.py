"""GEMM family kernels: GEMM, batched GEMM, grouped GEMM, tensor contraction.

The family is laid out per PLAN.md §2.1:

* :mod:`.reference` — float64 golden references defining numerical semantics
* :mod:`.impl`      — backend-specialized implementations (CPU today)
* :mod:`.tuning`    — auto-tuning configuration spaces
* :file:`KERNEL.md` — input/layout/dtype/perf-target contract
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ...runtime import dispatch


def gemm(a: np.ndarray, b: np.ndarray, c=None, alpha: float = 1.0, beta: float = 0.0):
    """C = alpha * (A @ B) + beta * C.

    A is (M, K), B is (K, N). Reduced-precision inputs accumulate in FP32.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"gemm expects 2D operands, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"gemm inner-dim mismatch: {a.shape} x {b.shape}")
    return dispatch("gemm", a, b, c=c, alpha=alpha, beta=beta)


def batched_gemm(a: np.ndarray, b: np.ndarray, alpha: float = 1.0):
    """Uniform-shape batched matmul. A is (..., M, K), B is (..., K, N)."""
    if a.ndim < 3 or b.ndim < 3:
        raise ValueError("batched_gemm expects >=3D operands")
    return dispatch("batched_gemm", a, b, alpha=alpha)


def grouped_gemm(
    a_list: Sequence[np.ndarray], b_list: Sequence[np.ndarray], alpha: float = 1.0
) -> List[np.ndarray]:
    """Variable-shape grouped matmul — the MoE expert-matmul primitive."""
    return dispatch("grouped_gemm", list(a_list), list(b_list), alpha=alpha)


def tensor_contraction(a: np.ndarray, b: np.ndarray, subscripts: str):
    """Generalized einsum-style contraction over the GEMM core."""
    return dispatch("tensor_contraction", a, b, subscripts)


__all__ = ["gemm", "batched_gemm", "grouped_gemm", "tensor_contraction"]
