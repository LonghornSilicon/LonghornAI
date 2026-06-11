"""Minimal auto-tuning driver.

Explores a discrete configuration space, benchmarks a parameterized kernel
factory at each point, and returns the best config keyed by measured latency.
In M1 the "space" is exhaustive grid search; the same interface accepts smarter
search strategies (coordinate descent, model-guided) later. Winning configs are
intended to be cached per (op, shape, dtype, backend) and fed to dispatch
(PLAN.md §6.1).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class TuningSpace:
    """A grid of named parameters to search over."""

    params: Dict[str, List[Any]] = field(default_factory=dict)

    def configs(self) -> List[Dict[str, Any]]:
        if not self.params:
            return [{}]
        keys = list(self.params)
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.params.values())]


@dataclass
class TuningResult:
    best_config: Dict[str, Any]
    best_latency_s: float
    all_results: List[Tuple[Dict[str, Any], float]]


def autotune(
    kernel_factory: Callable[..., Callable[[], Any]],
    space: TuningSpace,
    *,
    warmup: int = 2,
    trials: int = 10,
) -> TuningResult:
    """Grid-search ``space``; return the lowest-latency config.

    ``kernel_factory(**config)`` must return a zero-arg callable that runs the
    kernel for that configuration.
    """
    results: List[Tuple[Dict[str, Any], float]] = []
    for config in space.configs():
        run = kernel_factory(**config)
        for _ in range(max(0, warmup)):
            run()
        best = float("inf")
        for _ in range(max(1, trials)):
            t0 = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - t0)
        results.append((config, best))
    best_config, best_latency = min(results, key=lambda kv: kv[1])
    return TuningResult(best_config=best_config, best_latency_s=best_latency, all_results=results)
