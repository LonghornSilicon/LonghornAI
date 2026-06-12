"""Float64 references for the collectives family.

Collectives are structural — they re-arrange and reduce values across a
list of per-rank shards. The references encode the math in float64 so
the differential harness can validate that every backend (CPU shim today,
NCCL-equivalent on lhsil tomorrow) produces the same per-rank shards.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _f64_list(shards: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(s, dtype=np.float64) for s in shards]


def _reduce(shards: List[np.ndarray], op: str) -> np.ndarray:
    arr = np.stack(_f64_list(shards), axis=0)
    if op == "sum":
        return arr.sum(axis=0)
    if op == "max":
        return arr.max(axis=0)
    if op == "mean":
        return arr.mean(axis=0)
    raise ValueError(f"unknown reduction op '{op}'")


def ref_all_reduce(shards, *, op="sum"):
    out_dt = shards[0].dtype
    reduced = _reduce(shards, op).astype(out_dt)
    return [reduced.copy() for _ in shards]


def ref_all_gather(shards, *, axis=0):
    out_dt = shards[0].dtype
    full = np.concatenate([s.astype(np.float64) for s in shards], axis=axis).astype(out_dt)
    return [full.copy() for _ in shards]


def ref_reduce_scatter(shards, *, op="sum", axis=0):
    R = len(shards)
    reduced = _reduce(shards, op)
    if reduced.shape[axis] % R != 0:
        raise ValueError(
            f"reduce_scatter: axis-{axis} dim {reduced.shape[axis]} not "
            f"divisible by {R} ranks"
        )
    chunks = np.split(reduced, R, axis=axis)
    out_dt = shards[0].dtype
    return [c.astype(out_dt) for c in chunks]


def ref_all_to_all(shards, *, axis=0):
    R = len(shards)
    if shards[0].shape[axis] % R != 0:
        raise ValueError(
            f"all_to_all: axis-{axis} dim {shards[0].shape[axis]} not "
            f"divisible by {R} ranks"
        )
    # Split each rank's shard into R equal chunks along ``axis``; collect
    # the j-th chunk from every rank to form rank j's output.
    split = [np.split(s, R, axis=axis) for s in shards]   # split[i][j] = chunk
    out = []
    for j in range(R):
        out.append(np.concatenate([split[i][j] for i in range(R)], axis=axis))
    return out


REFERENCES = {
    "all_reduce": ref_all_reduce,
    "all_gather": ref_all_gather,
    "reduce_scatter": ref_reduce_scatter,
    "all_to_all": ref_all_to_all,
}

__all__ = [
    "REFERENCES",
    "ref_all_reduce",
    "ref_all_gather",
    "ref_reduce_scatter",
    "ref_all_to_all",
]
