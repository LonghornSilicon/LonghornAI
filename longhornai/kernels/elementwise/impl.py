"""CPU (NumPy) backend implementations for the elementwise / sequence family."""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu


@cpu.register("softmax")
def softmax(x, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    m = np.max(xa, axis=axis, keepdims=True)
    e = np.exp(xa - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(out_dt)


@cpu.register("rope")
def rope(x, positions=None, theta=10000.0):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    seq = xa.shape[-2]
    head_dim = xa.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("rope: head_dim must be even")
    half = head_dim // 2
    if positions is None:
        positions = np.arange(seq)
    pos = np.asarray(positions, dtype=acc)
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=acc) * 2.0 / head_dim))
    freqs = np.outer(pos, inv_freq)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    x1 = xa[..., :half]
    x2 = xa[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return np.concatenate([rot1, rot2], axis=-1).astype(out_dt)


@cpu.register("embedding_lookup")
def embedding_lookup(table, ids, scale=1.0):
    out = table[np.asarray(ids)]
    if scale != 1.0:
        out = (out.astype(_acc(out.dtype)) * scale).astype(table.dtype)
    return out


@cpu.register("reduce")
def reduce(x, op="sum", axis=-1, keepdims=False):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    if op == "sum":
        out = np.sum(xa, axis=axis, keepdims=keepdims)
    elif op == "max":
        out = np.max(xa, axis=axis, keepdims=keepdims)
    elif op == "mean":
        out = np.mean(xa, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"unknown reduce op '{op}'")
    return out.astype(out_dt)
