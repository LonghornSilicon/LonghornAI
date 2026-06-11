"""Benchmarking harness (PLAN.md §4).

Measures steady-state latency with warm-up, then derives the plan's metrics:
throughput, achieved FLOPS, achieved memory bandwidth, arithmetic intensity, and
roofline efficiency against a configurable device peak. Per-kernel FLOP/byte
cost models live in :mod:`longhornai.benchmarks.cost`.
"""

from __future__ import annotations

from .cost import CostModel, cost_for
from .harness import BenchmarkResult, DevicePeak, benchmark, roofline_efficiency

__all__ = [
    "benchmark",
    "BenchmarkResult",
    "DevicePeak",
    "roofline_efficiency",
    "CostModel",
    "cost_for",
]
