"""CPU implementations for the collectives family.

The CPU backend simulates a multi-rank execution by computing the
collective math directly on the supplied shard list. Real hardware
backends (lhsil) replace these with NCCL/RCCL/SHIM calls; the contract
— per-rank output shards — stays identical.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import cpu
from .reference import (
    ref_all_gather,
    ref_all_reduce,
    ref_all_to_all,
    ref_reduce_scatter,
)


@cpu.register("all_reduce")
def all_reduce(shards, *, op="sum"):
    return ref_all_reduce(shards, op=op)


@cpu.register("all_gather")
def all_gather(shards, *, axis=0):
    return ref_all_gather(shards, axis=axis)


@cpu.register("reduce_scatter")
def reduce_scatter(shards, *, op="sum", axis=0):
    return ref_reduce_scatter(shards, op=op, axis=axis)


@cpu.register("all_to_all")
def all_to_all(shards, *, axis=0):
    return ref_all_to_all(shards, axis=axis)
