"""Quantization primitives: symmetric integer quantize / dequantize.

Foundation for the Phase-4 quantization kernels. Symmetric, round-to-nearest
INT quantization with per-tensor or per-axis scales; this is the math AWQ/GPTQ/
SmoothQuant layer their calibration on top of.
"""

from __future__ import annotations

import numpy as np


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
