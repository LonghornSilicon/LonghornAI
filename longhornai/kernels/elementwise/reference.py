"""Float64 golden references for the elementwise / sequence family."""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


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


REFERENCES = {
    "softmax": ref_softmax,
    "rope": ref_rope,
    "embedding_lookup": ref_embedding_lookup,
    "reduce": ref_reduce,
}

__all__ = [
    "REFERENCES",
    "ref_softmax",
    "ref_rope",
    "ref_embedding_lookup",
    "ref_reduce",
]
