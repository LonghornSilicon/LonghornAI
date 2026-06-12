"""Collectives kernel family (PLAN.md §3 Phase 6 / §8 M6).

The four collectives that every multi-GPU LLM stack relies on:

* ``all_reduce``       — reduce-and-broadcast (sum / max / mean).
* ``all_gather``       — concatenate every rank's shard along an axis.
* ``reduce_scatter``   — reduce, then partition the result across ranks.
* ``all_to_all``       — each rank's shard is split into N chunks; chunk
  ``j`` from rank ``i`` becomes chunk ``i`` on rank ``j``. The token-
  routing primitive for expert parallelism.

These kernels operate on a **list of per-rank shards** — one ``np.ndarray``
per virtual rank. The CPU reference computes the math directly; lhsil's
backend lowering swaps in NCCL/RCCL/SHIM equivalents. The contract is the
*output* — every backend must return the same shards a real all-reduce on
a real interconnect would.

PLAN.md §8 M6 exit gate (TP + expert-parallel functional) consumes these
kernels via :mod:`longhornai.runtime.tensor_parallel` and
:mod:`longhornai.runtime.expert_parallel`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ...runtime import dispatch


def all_reduce(
    shards: List[np.ndarray], *, op: str = "sum",
) -> List[np.ndarray]:
    """Reduce ``shards`` across ranks; every rank receives the reduced result.

    ``op`` ∈ {"sum", "max", "mean"}. Every shard must share shape and dtype.
    """
    return dispatch("all_reduce", shards, op=op)


def all_gather(
    shards: List[np.ndarray], *, axis: int = 0,
) -> List[np.ndarray]:
    """Concatenate per-rank shards along ``axis``; every rank gets the full result.

    Shapes must match on every axis except ``axis``.
    """
    return dispatch("all_gather", shards, axis=axis)


def reduce_scatter(
    shards: List[np.ndarray], *, op: str = "sum", axis: int = 0,
) -> List[np.ndarray]:
    """All-reduce, then partition the reduced tensor across ranks along ``axis``.

    Each input shard already represents that rank's contribution to the
    full tensor (same shape on every rank); after the call rank ``r`` holds
    its 1/R slice along ``axis``. The reduced full-axis dim must be
    divisible by the number of ranks.
    """
    return dispatch("reduce_scatter", shards, op=op, axis=axis)


def all_to_all(
    shards: List[np.ndarray], *, axis: int = 0,
) -> List[np.ndarray]:
    """All-to-all exchange of equal-sized chunks along ``axis``.

    Each input shard's ``axis`` dim is split into ``num_ranks`` chunks;
    chunk ``j`` of rank ``i``'s input goes to rank ``j`` as its ``i``-th
    chunk in the output. Shapes match on every other axis. The ``axis``
    dim must be divisible by the number of ranks.
    """
    return dispatch("all_to_all", shards, axis=axis)


__all__ = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
]
