"""Quantization primitives: symmetric / asymmetric, groupwise, INT4 packing.

M1 shipped symmetric per-channel INT8. M5 (PLAN.md §3 Phase 4) extends:

* **Symmetric / asymmetric** — asymmetric carries a zero-point so the
  representable range is shifted to fit one-sided activations.
* **Per-tensor / per-channel / groupwise** — granularity controls how many
  scales are derived per quantized tensor. AWQ and GPTQ ship groupwise
  scales (group_size 32–128 per K) for INT4 weights.
* **INT4 packing** — two int4 values per byte. Storage is ``int8`` carrier
  with values in ``[-8, 7]``; ``pack_int4`` halves the K-axis, ``unpack_int4``
  restores it. Pack/unpack is bit-exact and round-trippable.

These primitives back the ``W4A16`` / ``W8A8`` GEMM kernels in
``kernels/quant/`` and the SmoothQuant / AWQ / GPTQ calibration algorithms
in ``quantization/calibration/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# --- M1 surface (kept for back-compat) --------------------------------------

def scales_per_channel(x: np.ndarray, bits: int = 8, axis: int = 0) -> np.ndarray:
    """Per-channel symmetric scale = max(|x|) / qmax along ``axis``."""
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    return np.maximum(amax / qmax, 1e-12).astype(np.float32)


def quantize(x: np.ndarray, scale, bits: int = 8) -> np.ndarray:
    """Symmetric quantize to signed ``bits``-wide integers."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    q = np.rint(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, qmin, qmax).astype(np.int32)


def dequantize(q: np.ndarray, scale) -> np.ndarray:
    """Inverse of :func:`quantize`."""
    return (q.astype(np.float64) * scale).astype(np.float32)


def quantize_dequantize(x: np.ndarray, bits: int = 8, axis: int = 0) -> np.ndarray:
    """Round-trip quant->dequant; the accuracy proxy used in calibration tests."""
    scale = scales_per_channel(x, bits=bits, axis=axis)
    return dequantize(quantize(x, scale, bits=bits), scale)


# --- M5: asymmetric quantization --------------------------------------------

@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for one tensor.

    ``scale`` and ``zero_point`` shape encodes granularity:
    * ``()``  — per-tensor
    * ``(1, ..., C, ..., 1)`` — per-channel along axis ``C``
    * ``(K // group_size, N)`` — groupwise (K is the reduction axis)
    """

    scale: np.ndarray              # float32
    zero_point: Optional[np.ndarray] = None  # int32, None → symmetric
    bits: int = 8
    granularity: str = "per_tensor"
    group_size: Optional[int] = None
    axis: Optional[int] = None


def _qrange(bits: int, asymmetric: bool) -> Tuple[int, int]:
    if asymmetric:
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def quantize_asymmetric(
    x: np.ndarray, *, bits: int = 8, axis: Optional[int] = None,
) -> Tuple[np.ndarray, QuantParams]:
    """Asymmetric (zero-point) quantization.

    Maps ``[xmin, xmax]`` linearly to ``[0, 2^bits - 1]``. ``axis`` selects
    per-channel along that axis; ``None`` → per-tensor.

    The zero-point is *not* clipped to ``[qmin, qmax]`` — it's an offset
    that maps ``q=qmin`` back to ``xmin`` via ``q = round(x/scale) + zp``.
    For one-sided positive data ``zp`` is negative; the *quantized values*
    themselves still land in ``[qmin, qmax]``.
    """
    qmin, qmax = _qrange(bits, asymmetric=True)
    if axis is None:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        scale = max((xmax - xmin) / (qmax - qmin), 1e-12)
        zp = qmin - round(xmin / scale)
        scale_arr = np.asarray(scale, dtype=np.float32)
        zp_arr = np.asarray(zp, dtype=np.int32)
    else:
        keep = [i for i in range(x.ndim) if i != axis]
        xmin = np.min(x, axis=tuple(keep), keepdims=True)
        xmax = np.max(x, axis=tuple(keep), keepdims=True)
        scale = np.maximum((xmax - xmin) / (qmax - qmin), 1e-12).astype(np.float32)
        zp = (qmin - np.round(xmin / scale)).astype(np.int32)
        scale_arr = scale.squeeze().astype(np.float32)
        zp_arr = zp.squeeze().astype(np.int32)
        # Restore broadcast shape against `x`
        shape = [1] * x.ndim
        shape[axis] = x.shape[axis]
        scale_arr = scale_arr.reshape(shape)
        zp_arr = zp_arr.reshape(shape)

    q = np.rint(x.astype(np.float64) / scale_arr) + zp_arr
    q = np.clip(q, qmin, qmax).astype(np.int32)
    return q, QuantParams(
        scale=scale_arr, zero_point=zp_arr, bits=bits,
        granularity="per_tensor" if axis is None else "per_channel",
        axis=axis,
    )


def quantize_groupwise(
    x: np.ndarray, *, bits: int = 4, group_size: int = 64, axis: int = 0,
    asymmetric: bool = False,
) -> Tuple[np.ndarray, QuantParams]:
    """Groupwise quantization along ``axis`` with chunks of ``group_size``.

    Standard for INT4 weights (W4A16): K is the reduction axis, groups of
    32–128 rows share one scale (and zero-point) per output column.
    """
    if x.shape[axis] % group_size != 0:
        raise ValueError(
            f"axis-{axis} dim ({x.shape[axis]}) must be divisible by "
            f"group_size ({group_size})"
        )
    qmin, qmax = _qrange(bits, asymmetric=asymmetric)
    # Move reduction axis to position 0, then split into groups.
    moved = np.moveaxis(x, axis, 0)        # (K, N0, N1, ...)
    K = moved.shape[0]
    n_groups = K // group_size
    grouped = moved.reshape(n_groups, group_size, *moved.shape[1:])
    # Compute per-group scale (and zero-point if asymmetric).
    if asymmetric:
        gmin = grouped.min(axis=1, keepdims=False)
        gmax = grouped.max(axis=1, keepdims=False)
        scale = np.maximum((gmax - gmin) / (qmax - qmin), 1e-12).astype(np.float32)
        zp = (qmin - np.round(gmin / scale)).clip(qmin, qmax).astype(np.int32)
        # Broadcast scale/zp back over the group dim
        scale_b = np.broadcast_to(scale[:, None, ...], grouped.shape)
        zp_b = np.broadcast_to(zp[:, None, ...], grouped.shape)
        q_grouped = (np.rint(grouped.astype(np.float64) / scale_b) + zp_b).clip(qmin, qmax).astype(np.int32)
    else:
        amax = np.maximum(np.abs(grouped).max(axis=1, keepdims=False), 1e-12)
        scale = (amax / (qmax)).astype(np.float32)
        scale_b = np.broadcast_to(scale[:, None, ...], grouped.shape)
        zp = None
        q_grouped = np.rint(grouped.astype(np.float64) / scale_b).clip(qmin, qmax).astype(np.int32)
    # Restore original axis order.
    q_flat = q_grouped.reshape(moved.shape)
    q = np.moveaxis(q_flat, 0, axis)
    params = QuantParams(
        scale=scale, zero_point=zp, bits=bits,
        granularity="group", group_size=group_size, axis=axis,
    )
    return q, params


def dequantize_q(q: np.ndarray, params: QuantParams) -> np.ndarray:
    """Inverse of :func:`quantize_asymmetric` / :func:`quantize_groupwise`."""
    if params.granularity == "group":
        moved = np.moveaxis(q, params.axis, 0)
        n_groups = moved.shape[0] // params.group_size
        grouped = moved.reshape(n_groups, params.group_size, *moved.shape[1:])
        scale_b = np.broadcast_to(params.scale[:, None, ...], grouped.shape)
        if params.zero_point is not None:
            zp_b = np.broadcast_to(params.zero_point[:, None, ...], grouped.shape)
            x_grouped = (grouped.astype(np.float64) - zp_b) * scale_b
        else:
            x_grouped = grouped.astype(np.float64) * scale_b
        x_flat = x_grouped.reshape(moved.shape)
        return np.moveaxis(x_flat, 0, params.axis).astype(np.float32)
    if params.zero_point is not None:
        return ((q.astype(np.float64) - params.zero_point) * params.scale).astype(np.float32)
    return (q.astype(np.float64) * params.scale).astype(np.float32)


# --- INT4 packing -----------------------------------------------------------
# Two signed int4 values per int8 byte. Low nibble first.

def pack_int4(q: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Pack int4 values (stored as int8 in [-8, 7]) two per byte along ``axis``.

    Output dtype is int8; the ``axis`` dim halves. Even-axis-length only.
    """
    if q.shape[axis] % 2 != 0:
        raise ValueError(f"pack_int4 requires axis-{axis} dim to be even")
    moved = np.moveaxis(q, axis, 0)             # (K, ...)
    K = moved.shape[0]
    low = moved[0::2].astype(np.int32) & 0x0F   # bottom 4 bits
    high = moved[1::2].astype(np.int32) & 0x0F
    packed = (low | (high << 4)).astype(np.uint8)
    # Re-interpret as int8 carrier (same bit pattern; readers must mask).
    packed = packed.view(np.int8)
    return np.moveaxis(packed, 0, axis)


def unpack_int4(packed: np.ndarray, *, axis: int = 0, signed: bool = True) -> np.ndarray:
    """Inverse of :func:`pack_int4`. Returns int8 with values in [-8, 7] (signed)."""
    moved = np.moveaxis(packed, axis, 0).view(np.uint8)
    K = moved.shape[0]
    out = np.empty((K * 2, *moved.shape[1:]), dtype=np.int8)
    low = (moved & 0x0F).astype(np.int8)
    high = ((moved >> 4) & 0x0F).astype(np.int8)
    if signed:
        # Sign-extend 4-bit two's complement.
        low = np.where(low >= 8, low - 16, low).astype(np.int8)
        high = np.where(high >= 8, high - 16, high).astype(np.int8)
    out[0::2] = low
    out[1::2] = high
    return np.moveaxis(out, 0, axis)


__all__ = [
    # M1 surface
    "scales_per_channel",
    "quantize",
    "dequantize",
    "quantize_dequantize",
    # M5 surface
    "QuantParams",
    "quantize_asymmetric",
    "quantize_groupwise",
    "dequantize_q",
    "pack_int4",
    "unpack_int4",
]
