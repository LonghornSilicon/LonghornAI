"""Float64 references for the quantized-GEMM family.

The reference computes the *mathematical* output of a quantized GEMM —
INT × INT → INT32 accumulate → scale to float — in float64 for parity
checks. The W4A16 reference dequantizes weights via the groupwise scales,
then runs a dense matmul.
"""

from __future__ import annotations

import numpy as np

from ..gemm.reference import _f64
from ...quantization.primitives import unpack_int4


def ref_gemm_w8a8(a_q, b_q, *, scale_a, scale_b,
                  zero_point_a=None, zero_point_b=None,
                  out_dtype=np.float16):
    """INT8 × INT8 GEMM with INT32 accumulation, then float-scale."""
    a_i = a_q.astype(np.int32)
    b_i = b_q.astype(np.int32)
    if zero_point_a is not None:
        a_i = a_i - zero_point_a.astype(np.int32)
    if zero_point_b is not None:
        b_i = b_i - zero_point_b.astype(np.int32)
    prod = a_i @ b_i                                       # int32
    scaled = prod.astype(np.float64) * np.asarray(scale_a, dtype=np.float64) \
                                     * np.asarray(scale_b, dtype=np.float64)
    return scaled.astype(out_dtype)


def ref_gemm_w4a16(a, b_q_packed, *, scale_b, zero_point_b=None,
                   group_size, K, out_dtype=None):
    """Dequantize weights via groupwise scales, then run dense matmul."""
    b_q = unpack_int4(b_q_packed, axis=0, signed=True)     # (K, N) int8
    # Groupwise dequant: scale_b is (K // group_size, N).
    n_groups = K // group_size
    b_grouped = b_q.reshape(n_groups, group_size, -1).astype(np.float64)
    scale_g = scale_b[:, None, :].astype(np.float64)
    if zero_point_b is not None:
        zp = zero_point_b[:, None, :].astype(np.float64)
        b_dq = (b_grouped - zp) * scale_g
    else:
        b_dq = b_grouped * scale_g
    b_dq = b_dq.reshape(K, -1)
    out = _f64(a) @ b_dq
    if out_dtype is None:
        out_dtype = a.dtype
    return out.astype(out_dtype)


def ref_activation_quantize(x, *, bits=8, axis=-1, asymmetric=False):
    """Dynamic per-row symmetric or asymmetric activation quantize.

    Returns ``(q, scale[, zero_point])``. The reference uses one scale per
    row along ``axis`` — i.e., per-token quant for a (B, S, D)-shaped
    activation with ``axis=-1``.
    """
    qmax = (1 << bits) - 1 if asymmetric else (1 << (bits - 1)) - 1
    qmin = 0 if asymmetric else -(1 << (bits - 1))
    if asymmetric:
        xmin = np.min(x, axis=axis, keepdims=True)
        xmax = np.max(x, axis=axis, keepdims=True)
        scale = np.maximum((xmax - xmin) / (qmax - qmin), 1e-12).astype(np.float32)
        zp = (qmin - np.round(xmin / scale)).astype(np.int32)
        q = np.clip(np.rint(x.astype(np.float64) / scale) + zp, qmin, qmax).astype(np.int32)
        return q, scale, zp
    amax = np.maximum(np.max(np.abs(x), axis=axis, keepdims=True), 1e-12)
    scale = (amax / qmax).astype(np.float32)
    q = np.clip(np.rint(x.astype(np.float64) / scale), qmin, qmax).astype(np.int32)
    return q, scale, None


def ref_activation_dequantize(q, *, scale, zero_point=None, dtype=np.float16):
    if zero_point is not None:
        return ((q.astype(np.float64) - zero_point) * scale).astype(dtype)
    return (q.astype(np.float64) * scale).astype(dtype)


REFERENCES = {
    "gemm_w8a8": ref_gemm_w8a8,
    "gemm_w4a16": ref_gemm_w4a16,
    # activation quant returns a tuple; differential_test concatenates flat,
    # which works for the (q, scale[, zp]) tuple after we strip the None.
}

__all__ = [
    "REFERENCES",
    "ref_gemm_w8a8",
    "ref_gemm_w4a16",
    "ref_activation_quantize",
    "ref_activation_dequantize",
]
