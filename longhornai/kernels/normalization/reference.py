"""Float64 golden references for the normalization family."""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def ref_layernorm(x, weight=None, bias=None, eps=1e-5, axis=-1):
    x = _f64(x)
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * _f64(weight)
    if bias is not None:
        out = out + _f64(bias)
    return out


def ref_rmsnorm(x, weight=None, eps=1e-6, axis=-1):
    x = _f64(x)
    ms = np.mean(np.square(x), axis=axis, keepdims=True)
    out = x / np.sqrt(ms + eps)
    if weight is not None:
        out = out * _f64(weight)
    return out


REFERENCES = {"layernorm": ref_layernorm, "rmsnorm": ref_rmsnorm}

__all__ = ["REFERENCES", "ref_layernorm", "ref_rmsnorm"]
