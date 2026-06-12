"""CPU (NumPy) implementations for the quantized-GEMM family.

The CPU impl matches the math of the reference — int32 accumulation for
W8A8, dequant-then-matmul for W4A16. The point is *correctness*, not
performance; lhsil's W4A16 kernel will fuse the unpack + dequant into a
single tensor-core launch.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import cpu
from ...quantization.primitives import unpack_int4


@cpu.register("gemm_w8a8")
def gemm_w8a8(a_q, b_q, *, scale_a, scale_b,
              zero_point_a=None, zero_point_b=None,
              out_dtype=np.float16):
    a_i = a_q.astype(np.int32)
    b_i = b_q.astype(np.int32)
    if zero_point_a is not None:
        a_i = a_i - zero_point_a.astype(np.int32)
    if zero_point_b is not None:
        b_i = b_i - zero_point_b.astype(np.int32)
    prod = a_i @ b_i
    scaled = (prod.astype(np.float64)
              * np.asarray(scale_a, dtype=np.float64)
              * np.asarray(scale_b, dtype=np.float64))
    return scaled.astype(out_dtype)


@cpu.register("gemm_w4a16")
def gemm_w4a16(a, b_q_packed, *, scale_b, zero_point_b=None,
               group_size, K, out_dtype=None):
    b_q = unpack_int4(b_q_packed, axis=0, signed=True)         # (K, N)
    n_groups = K // group_size
    b_grouped = b_q.reshape(n_groups, group_size, -1).astype(np.float64)
    scale_g = scale_b[:, None, :].astype(np.float64)
    if zero_point_b is not None:
        zp = zero_point_b[:, None, :].astype(np.float64)
        b_dq = (b_grouped - zp) * scale_g
    else:
        b_dq = b_grouped * scale_g
    b_dq = b_dq.reshape(K, -1)
    out = a.astype(np.float64) @ b_dq
    if out_dtype is None:
        out_dtype = a.dtype
    return out.astype(out_dtype)


@cpu.register("activation_quantize")
def activation_quantize(x, *, bits=8, axis=-1, asymmetric=False):
    qmax = (1 << bits) - 1 if asymmetric else (1 << (bits - 1)) - 1
    qmin = 0 if asymmetric else -(1 << (bits - 1))
    if asymmetric:
        xmin = np.min(x, axis=axis, keepdims=True)
        xmax = np.max(x, axis=axis, keepdims=True)
        scale = np.maximum((xmax - xmin) / (qmax - qmin), 1e-12).astype(np.float32)
        # Standard asymmetric zero-point — must NOT be clipped to [qmin, qmax]
        # because zp represents the *offset* that maps q=0 back to xmin via
        # q = round(x/scale) + zp. With xmin > 0 zp is negative; the q
        # values themselves still land in [qmin, qmax].
        zp = (qmin - np.round(xmin / scale)).astype(np.int32)
        # Asymmetric INT8 carrier: uint8 (range [0, 255]).
        q = np.clip(np.rint(x.astype(np.float64) / scale) + zp, qmin, qmax).astype(np.uint8)
        return q, scale, zp
    amax = np.maximum(np.max(np.abs(x), axis=axis, keepdims=True), 1e-12)
    scale = (amax / qmax).astype(np.float32)
    q = np.clip(np.rint(x.astype(np.float64) / scale), qmin, qmax).astype(np.int8)
    return q, scale, None


@cpu.register("activation_dequantize")
def activation_dequantize(q, *, scale, zero_point=None, dtype=np.float16):
    if zero_point is not None:
        return ((q.astype(np.float64) - zero_point) * scale).astype(dtype)
    return (q.astype(np.float64) * scale).astype(dtype)
