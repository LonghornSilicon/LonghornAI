"""CPU (NumPy) backend implementations for the activation family."""

from __future__ import annotations

import math

import numpy as np

from ...backends.cpu import _acc, cpu


@cpu.register("gelu")
def gelu(x, approximate="none"):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        out = 0.5 * xa * (1.0 + np.tanh(c * (xa + 0.044715 * xa**3)))
    else:
        erf = np.vectorize(math.erf, otypes=[np.float64])
        out = 0.5 * xa * (1.0 + erf(xa / math.sqrt(2.0)).astype(acc))
    return out.astype(out_dt)


@cpu.register("silu")
def silu(x):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    return (xa / (1.0 + np.exp(-xa))).astype(out_dt)
