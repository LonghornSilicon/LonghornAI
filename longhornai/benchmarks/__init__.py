"""Benchmarking harness (PLAN.md §4).

Measures steady-state latency with warm-up, then derives the plan's metrics:
throughput, achieved FLOPS, achieved memory bandwidth, arithmetic intensity, and
roofline efficiency against a configurable device peak. Per-kernel FLOP/byte
cost models live in :mod:`longhornai.benchmarks.cost`.
"""

from __future__ import annotations

from .cost import CostModel, cost_for
from .decode_bench import (
    DecodeBenchmarkResult,
    load_baseline as load_decode_baseline,
    run_decode_benchmark,
    save_baseline as save_decode_baseline,
)
from .harness import BenchmarkResult, DevicePeak, benchmark, roofline_efficiency
from .shape_suite import (
    GEMM_SHAPE_SUITE,
    ShapeCase,
    ShapeResult,
    compare_to_baseline,
    format_suite_report,
    load_baseline,
    run_gemm_suite,
    save_baseline,
)

__all__ = [
    "benchmark",
    "BenchmarkResult",
    "DevicePeak",
    "roofline_efficiency",
    "CostModel",
    "cost_for",
    "ShapeCase",
    "ShapeResult",
    "GEMM_SHAPE_SUITE",
    "run_gemm_suite",
    "format_suite_report",
    "load_baseline",
    "save_baseline",
    "compare_to_baseline",
    "DecodeBenchmarkResult",
    "run_decode_benchmark",
    "load_decode_baseline",
    "save_decode_baseline",
]
