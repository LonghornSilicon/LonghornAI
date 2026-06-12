"""Quantized GEMM family (PLAN.md §3 Phase 4 / M5).

Operators:

* ``gemm_w8a8`` — INT8 weights × INT8 activations, INT32 accumulate, FP32
  scale to FP-output. The W8A8 perf path; SmoothQuant rescues accuracy
  when activations have outliers (PLAN.md §3 Phase 4).
* ``gemm_w4a16`` — INT4 *packed* weights × FP16 activations, FP32 accumulate
  via dequant-on-the-fly. Groupwise scales (group_size 32–128) along the
  reduction axis. The dominant LLM-decode inference path because it cuts
  weight bandwidth ~4×.
* ``activation_quantize`` / ``activation_dequantize`` — dynamic activation
  quant utilities for fused W8A8 prologues / epilogues.

PLAN.md §8 M5 exit gate: INT8 W8A8 ≤ 1% perplexity delta vs FP16; INT4
W4A16 ≤ defined per-model bound — both at ≥ measured target speedup.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...runtime import dispatch


def gemm_w8a8(
    a_q: np.ndarray,
    b_q: np.ndarray,
    *,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    zero_point_a: Optional[np.ndarray] = None,
    zero_point_b: Optional[np.ndarray] = None,
    out_dtype=np.float16,
) -> np.ndarray:
    """INT8 weights × INT8 activations GEMM with INT32 accumulation.

    ``a_q``: int8 ``(M, K)``. ``b_q``: int8 ``(K, N)``.
    ``scale_a``: per-tensor scalar OR per-row ``(M, 1)`` float32.
    ``scale_b``: per-tensor scalar OR per-column ``(1, N)`` float32.
    Asymmetric is supported via the optional ``zero_point_*`` args.
    """
    return dispatch(
        "gemm_w8a8", a_q, b_q,
        scale_a=scale_a, scale_b=scale_b,
        zero_point_a=zero_point_a, zero_point_b=zero_point_b,
        out_dtype=out_dtype,
    )


def gemm_w4a16(
    a: np.ndarray,
    b_q_packed: np.ndarray,
    *,
    scale_b: np.ndarray,
    zero_point_b: Optional[np.ndarray] = None,
    group_size: int,
    K: int,
    out_dtype=None,
) -> np.ndarray:
    """INT4 packed weights × FP16/FP32 activations GEMM.

    ``a``: ``(M, K)`` floating activations.
    ``b_q_packed``: int8 ``(K // 2, N)`` packed INT4 weights along axis 0.
    ``scale_b``: groupwise scales, ``(K // group_size, N)``, float32.
    ``zero_point_b`` (optional): groupwise zero-points (for asymmetric).
    ``K``: full unpacked reduction dim (the kernel needs it because ``K`` is
    halved in the packed carrier).
    """
    return dispatch(
        "gemm_w4a16", a, b_q_packed,
        scale_b=scale_b, zero_point_b=zero_point_b,
        group_size=group_size, K=K, out_dtype=out_dtype,
    )


def activation_quantize(
    x: np.ndarray, *, bits: int = 8, axis: int = -1, asymmetric: bool = False,
):
    """Dynamic activation quantize. Returns ``(q, scale[, zero_point])``."""
    return dispatch(
        "activation_quantize", x,
        bits=bits, axis=axis, asymmetric=asymmetric,
    )


def activation_dequantize(
    q: np.ndarray, *, scale: np.ndarray,
    zero_point: Optional[np.ndarray] = None, dtype=np.float16,
) -> np.ndarray:
    """Inverse of :func:`activation_quantize`."""
    return dispatch(
        "activation_dequantize", q,
        scale=scale, zero_point=zero_point, dtype=dtype,
    )


__all__ = [
    "gemm_w8a8",
    "gemm_w4a16",
    "activation_quantize",
    "activation_dequantize",
]
