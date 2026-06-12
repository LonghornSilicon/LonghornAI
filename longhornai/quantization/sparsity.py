"""Structured N:M sparsity (PLAN.md §3 Phase 6 / §8 M7).

NVIDIA's Ampere+ silicon and Longhorn's tensor cores support **2:4
structured sparsity** — within every 4-element group along the reduction
axis, exactly 2 weights are zero. The hardware skips those zeros, giving
~2× compute throughput on dense FP without changing numerics outside the
pruning step itself.

This module ships:

* :func:`prune_2to4` — per-group keep-2-of-4-largest pruning (or any N:M).
  Returns a dense tensor with the masked-out entries zeroed; lhsil's
  packing format strips the zeros at runtime.
* :func:`gemm_sparse_2to4` — sparse-tensor GEMM. The CPU reference is just
  a dense matmul on the pruned weight; lhsil's kernel issues the 2:4
  sparse-tensor-core instruction and gets the 2× throughput.

The accuracy contract is bounded by the pruning step only — at-runtime the
math is dense.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def prune_n_m(
    weight: np.ndarray, *, n: int = 2, m: int = 4, axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the top-``n`` magnitudes in every ``m``-element group along ``axis``.

    Returns ``(pruned_weight, mask)`` — same shape as input. ``mask`` is
    boolean: ``True`` where the value was kept.
    """
    if weight.shape[axis] % m != 0:
        raise ValueError(
            f"prune_n_m: axis-{axis} dim ({weight.shape[axis]}) not "
            f"divisible by m ({m})"
        )
    moved = np.moveaxis(weight, axis, 0)              # (K, ...)
    K = moved.shape[0]
    n_groups = K // m
    grouped = moved.reshape(n_groups, m, *moved.shape[1:])

    # Per-group threshold: pick the top-n absolute values per group.
    abs_g = np.abs(grouped)
    # argsort ascending; the largest n indices are the last n.
    sorted_idx = np.argsort(abs_g, axis=1)
    keep_idx = sorted_idx[:, -n:, ...]
    mask = np.zeros_like(grouped, dtype=bool)
    np.put_along_axis(mask, keep_idx, True, axis=1)
    pruned = np.where(mask, grouped, 0)

    pruned = np.moveaxis(pruned.reshape(moved.shape), 0, axis)
    mask = np.moveaxis(mask.reshape(moved.shape), 0, axis)
    return pruned, mask


def prune_2to4(
    weight: np.ndarray, *, axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: 2:4 structured pruning along ``axis``."""
    return prune_n_m(weight, n=2, m=4, axis=axis)


def gemm_sparse_2to4(
    a: np.ndarray, b_pruned: np.ndarray, *, axis: int = 0,
) -> np.ndarray:
    """Dense matmul against a 2:4-pruned weight.

    The CPU reference is `a @ b_pruned`; lhsil's sparse-tensor-core kernel
    skips the zeroed entries for ~2× throughput. Equivalent numerically.

    ``axis`` documents which axis of ``b_pruned`` was pruned (typically
    the reduction axis = 0); the dense matmul doesn't actually use it but
    backends may key their packing on this hint.
    """
    return np.asarray(a, dtype=np.float64) @ np.asarray(b_pruned, dtype=np.float64) \
        .astype(a.dtype if a.dtype.kind == "f" else np.float32)


__all__ = [
    "prune_n_m",
    "prune_2to4",
    "gemm_sparse_2to4",
]
