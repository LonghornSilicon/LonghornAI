"""Float64 golden references for the activation family."""

from __future__ import annotations

import math

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _erf_vec(x: np.ndarray) -> np.ndarray:
    # math.erf is scalar-only; vectorize without taking a SciPy dependency.
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def ref_gelu(x, approximate="none"):
    x = _f64(x)
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))
    return 0.5 * x * (1.0 + _erf_vec(x / math.sqrt(2.0)))


def ref_gelu_exact(x):
    return ref_gelu(x, approximate="none")


def ref_silu(x):
    x = _f64(x)
    return x / (1.0 + np.exp(-x))


REFERENCES = {
    "gelu": ref_gelu_exact,
    "gelu_tanh": lambda x: ref_gelu(x, approximate="tanh"),
    "silu": ref_silu,
}

__all__ = ["REFERENCES", "ref_gelu", "ref_gelu_exact", "ref_silu"]
