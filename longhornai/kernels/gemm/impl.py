"""CPU (NumPy) backend implementations for the GEMM family.

The portable, always-available reference target. Accumulates in float64 so the
backend stays as accurate as the float64 golden — any divergence reported by
the differential harness is a real bug, not CPU-accumulation noise (PLAN.md
§2.2 backend abstraction; §5.1 numerical validation).
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu


@cpu.register("gemm")
def gemm(a, b, c=None, alpha=1.0, beta=0.0):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    res = alpha * (a.astype(acc) @ b.astype(acc))
    if c is not None and beta != 0.0:
        res = res + beta * np.asarray(c, dtype=acc)
    return res.astype(out_dt)


@cpu.register("batched_gemm")
def batched_gemm(a, b, alpha=1.0):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    return (alpha * (a.astype(acc) @ b.astype(acc))).astype(out_dt)


@cpu.register("grouped_gemm")
def grouped_gemm(a_list, b_list, alpha=1.0):
    if len(a_list) != len(b_list):
        raise ValueError("grouped_gemm: a_list and b_list length mismatch")
    return [gemm(a, b, alpha=alpha) for a, b in zip(a_list, b_list)]


@cpu.register("tensor_contraction")
def tensor_contraction(a, b, subscripts):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    return np.einsum(subscripts, a.astype(acc), b.astype(acc)).astype(out_dt)
