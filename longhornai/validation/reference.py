"""Float64 golden references — the numerical contract for each operator.

These implementations define *what each kernel must compute*, in full double
precision, with no concern for performance. Every backend kernel is validated
against the matching reference here (PLAN.md §5.1 golden references). When
semantics are ambiguous, the reference is authoritative.

All references take and return NumPy arrays in float64 and are pure functions.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np


def _f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --- GEMM family -------------------------------------------------------------

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


# --- normalization -----------------------------------------------------------

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


# --- activations -------------------------------------------------------------

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


# --- elementwise / sequence --------------------------------------------------

def ref_softmax(x, axis=-1):
    x = _f64(x)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def ref_rope(x, positions=None, theta=10000.0):
    """Rotary embedding on x of shape (..., seq, head_dim), head_dim even."""
    x = _f64(x)
    seq = x.shape[-2]
    head_dim = x.shape[-1]
    half = head_dim // 2
    if positions is None:
        positions = np.arange(seq)
    positions = _f64(positions)
    inv_freq = 1.0 / (theta ** (np.arange(0, half) * 2.0 / head_dim))
    freqs = np.outer(positions, inv_freq)  # (seq, half)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return np.concatenate([rot1, rot2], axis=-1)


def ref_embedding_lookup(table, ids, scale=1.0):
    out = _f64(table)[np.asarray(ids)]
    return out * scale


def ref_reduce(x, op="sum", axis=-1, keepdims=False):
    x = _f64(x)
    if op == "sum":
        return np.sum(x, axis=axis, keepdims=keepdims)
    if op == "max":
        return np.max(x, axis=axis, keepdims=keepdims)
    if op == "mean":
        return np.mean(x, axis=axis, keepdims=keepdims)
    raise ValueError(f"unknown reduce op '{op}'")


# Registry: op_name -> golden reference. Differential tests iterate this.
REFERENCES: Dict[str, Callable] = {
    "gemm": ref_gemm,
    "batched_gemm": ref_batched_gemm,
    "grouped_gemm": ref_grouped_gemm,
    "tensor_contraction": ref_tensor_contraction,
    "layernorm": ref_layernorm,
    "rmsnorm": ref_rmsnorm,
    "gelu": ref_gelu_exact,
    "gelu_tanh": lambda x: ref_gelu(x, approximate="tanh"),
    "silu": ref_silu,
    "softmax": ref_softmax,
    "rope": ref_rope,
    "embedding_lookup": ref_embedding_lookup,
    "reduce": ref_reduce,
}

__all__: List[str] = ["REFERENCES"] + [n for n in dir() if n.startswith("ref_")]
