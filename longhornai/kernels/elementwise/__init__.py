"""Elementwise / sequence kernels: softmax, RoPE, embedding lookup, reductions.

Family layout per PLAN.md §2.1.
"""

from __future__ import annotations

import numpy as np

from ...runtime import dispatch


def softmax(x: np.ndarray, axis: int = -1):
    """Numerically stable softmax over ``axis``."""
    return dispatch("softmax", x, axis=axis)


def rope(x: np.ndarray, positions=None, theta: float = 10000.0):
    """Rotary position embedding on x of shape (..., seq, head_dim).

    ``head_dim`` must be even. ``positions`` defaults to ``arange(seq)``.
    """
    return dispatch("rope", x, positions=positions, theta=theta)


def embedding_lookup(table: np.ndarray, ids, scale: float = 1.0):
    """Gather rows of ``table`` indexed by integer ``ids``, optionally scaled."""
    return dispatch("embedding_lookup", table, ids, scale=scale)


def reduce(x: np.ndarray, op: str = "sum", axis: int = -1, keepdims: bool = False):
    """Reduction primitive: ``op`` in {"sum", "max", "mean"}."""
    return dispatch("reduce", x, op=op, axis=axis, keepdims=keepdims)


__all__ = ["softmax", "rope", "embedding_lookup", "reduce"]
