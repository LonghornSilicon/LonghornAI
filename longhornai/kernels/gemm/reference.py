"""Float64 golden references for the GEMM family.

These define the numerical contract for ``gemm`` and friends. The differential
test harness aggregates them into :data:`longhornai.validation.REFERENCES` and
asserts every backend kernel agrees within the per-dtype tolerance policy.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def ref_gemm(a, b, c=None, alpha=1.0, beta=0.0):
    a, b = _f64(a), _f64(b)
    out = alpha * (a @ b)
    if c is not None and beta != 0.0:
        out = out + beta * _f64(c)
    return out


def ref_batched_gemm(a, b, alpha=1.0):
    return alpha * (_f64(a) @ _f64(b))


def ref_grouped_gemm(a_list, b_list, alpha=1.0):
    return [alpha * (_f64(a) @ _f64(b)) for a, b in zip(a_list, b_list)]


def ref_tensor_contraction(a, b, subscripts):
    return np.einsum(subscripts, _f64(a), _f64(b))


REFERENCES = {
    "gemm": ref_gemm,
    "batched_gemm": ref_batched_gemm,
    "grouped_gemm": ref_grouped_gemm,
    "tensor_contraction": ref_tensor_contraction,
}

__all__ = [
    "REFERENCES",
    "ref_gemm",
    "ref_batched_gemm",
    "ref_grouped_gemm",
    "ref_tensor_contraction",
]
