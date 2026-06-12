"""CPU (NumPy) backend implementations for the normalization family."""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu


@cpu.register("layernorm")
def layernorm(x, weight=None, bias=None, eps=1e-5, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    mean = xa.mean(axis=axis, keepdims=True)
    var = xa.var(axis=axis, keepdims=True)
    out = (xa - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight.astype(acc)
    if bias is not None:
        out = out + bias.astype(acc)
    return out.astype(out_dt)


@cpu.register("rmsnorm")
def rmsnorm(x, weight=None, eps=1e-6, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    ms = np.mean(np.square(xa), axis=axis, keepdims=True)
    out = xa / np.sqrt(ms + eps)
    if weight is not None:
        out = out * weight.astype(acc)
    return out.astype(out_dt)
