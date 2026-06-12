"""Persistent autotune cache (PLAN.md §6.1 / §8 M8).

PLAN.md §6.1 calls for "winning configs ... cached per (op, shape, dtype,
backend) and fed to dispatch". This module is that cache: a small JSON file
keyed by ``(op, shape_signature, dtype, backend)`` whose values are the best
configs the autotuner has found.

Workflow:

1. The first time a kernel is invoked at a new shape/dtype/backend, the
   tuner runs and persists the winning config.
2. Subsequent invocations look the config up by key and skip tuning.
3. On silicon bring-up the cache is regenerated; CI verifies that every
   shape in :mod:`longhornai.benchmarks.full_sweep` has an entry.

The cache is intentionally simple — JSON, in-tree default path. Production
silicon backends will swap in a more sophisticated key derivation (occupancy
bucket, register pressure tier) but the *interface* stays the same.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .autotune import TuningResult, TuningSpace, autotune


@dataclass(frozen=True)
class TuneKey:
    """One autotune cache entry's key."""

    op: str
    shape: Tuple[int, ...]
    dtype: str
    backend: str

    def to_json(self) -> str:
        return json.dumps({
            "op": self.op, "shape": list(self.shape),
            "dtype": self.dtype, "backend": self.backend,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "TuneKey":
        d = json.loads(s)
        return cls(op=d["op"], shape=tuple(d["shape"]),
                   dtype=d["dtype"], backend=d["backend"])


@dataclass
class TuneCache:
    """Disk-persisted (op, shape, dtype, backend) → best config map."""

    path: pathlib.Path
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | pathlib.Path) -> "TuneCache":
        p = pathlib.Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            return cls(path=p, entries=data.get("entries", {}))
        return cls(path=p)

    def lookup(self, key: TuneKey) -> Optional[Dict[str, Any]]:
        rec = self.entries.get(key.to_json())
        return rec["config"] if rec else None

    def store(self, key: TuneKey, config: Dict[str, Any], latency_s: float) -> None:
        self.entries[key.to_json()] = {
            "config": config,
            "latency_s": latency_s,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "entries": self.entries}
        self.path.write_text(json.dumps(payload, indent=2))

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> List[TuneKey]:
        return [TuneKey.from_json(k) for k in self.entries.keys()]


def autotune_cached(
    cache: TuneCache,
    key: TuneKey,
    *,
    kernel_factory: Callable[..., Callable[[], Any]],
    space: TuningSpace,
    warmup: int = 1,
    trials: int = 5,
    save_after: bool = True,
) -> Dict[str, Any]:
    """Look up ``key``; on miss, run autotune and persist the winning config."""
    cached = cache.lookup(key)
    if cached is not None:
        return cached
    result: TuningResult = autotune(
        kernel_factory, space, warmup=warmup, trials=trials,
    )
    cache.store(key, result.best_config, result.best_latency_s)
    if save_after:
        cache.save()
    return result.best_config


def shape_signature(*tensors: np.ndarray) -> Tuple[int, ...]:
    """Build a flat shape signature from tensor inputs for use as a cache key."""
    out: List[int] = []
    for t in tensors:
        out.extend(int(d) for d in t.shape)
    return tuple(out)


__all__ = [
    "TuneKey",
    "TuneCache",
    "autotune_cached",
    "shape_signature",
]
