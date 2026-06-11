"""Benchmark timing and roofline analysis.

Times a callable to steady state (warm-up + repeated trials, reporting the best
trial as the latency estimate to suppress scheduler noise), then combines the
measured latency with an analytic :class:`~longhornai.benchmarks.cost.CostModel`
to produce the PLAN.md §4.2 metric set, including roofline efficiency against a
configurable device peak.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .cost import CostModel


@dataclass(frozen=True)
class DevicePeak:
    """Peak compute and memory bandwidth of a target, for roofline analysis."""

    name: str
    peak_flops: float  # FLOP/s
    peak_bandwidth: float  # bytes/s

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline knee sits."""
        return self.peak_flops / self.peak_bandwidth


# A reasonable single-device reference peak for reporting (not target-specific).
# ~990 TFLOP/s FP16 dense, ~3.35 TB/s HBM — placeholder until lhsil specs land.
REFERENCE_PEAK = DevicePeak(
    name="reference",
    peak_flops=990e12,
    peak_bandwidth=3.35e12,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured + derived metrics for one benchmarked kernel."""

    name: str
    latency_s: float
    cost: CostModel
    peak: DevicePeak

    @property
    def throughput_ops_s(self) -> float:
        return 1.0 / self.latency_s if self.latency_s else float("inf")

    @property
    def achieved_flops(self) -> float:
        return self.cost.flops / self.latency_s

    @property
    def achieved_bandwidth(self) -> float:
        return self.cost.bytes_moved / self.latency_s

    @property
    def flops_utilization(self) -> float:
        return self.achieved_flops / self.peak.peak_flops

    @property
    def bandwidth_utilization(self) -> float:
        return self.achieved_bandwidth / self.peak.peak_bandwidth

    @property
    def is_compute_bound(self) -> bool:
        return self.cost.arithmetic_intensity >= self.peak.ridge_point

    @property
    def roofline_efficiency(self) -> float:
        """Achieved vs. the attainable performance at this arithmetic intensity."""
        attainable = min(
            self.peak.peak_flops,
            self.peak.peak_bandwidth * self.cost.arithmetic_intensity,
        )
        return self.achieved_flops / attainable

    def report(self) -> str:
        bound = "compute" if self.is_compute_bound else "memory"
        return (
            f"{self.name}\n"
            f"  latency        : {self.latency_s * 1e3:9.4f} ms\n"
            f"  throughput     : {self.throughput_ops_s:9.1f} ops/s\n"
            f"  achieved FLOPS : {self.achieved_flops / 1e12:9.3f} TFLOP/s "
            f"({self.flops_utilization * 100:5.1f}% of peak)\n"
            f"  achieved BW    : {self.achieved_bandwidth / 1e9:9.3f} GB/s "
            f"({self.bandwidth_utilization * 100:5.1f}% of peak)\n"
            f"  arith intensity: {self.cost.arithmetic_intensity:9.2f} FLOP/byte "
            f"({bound}-bound)\n"
            f"  roofline eff   : {self.roofline_efficiency * 100:9.1f}%"
        )


def roofline_efficiency(cost: CostModel, latency_s: float, peak: DevicePeak) -> float:
    """Standalone roofline efficiency for a cost/latency/peak triple."""
    achieved = cost.flops / latency_s
    attainable = min(peak.peak_flops, peak.peak_bandwidth * cost.arithmetic_intensity)
    return achieved / attainable


def benchmark(
    fn,
    cost: CostModel,
    *,
    name: str = "kernel",
    warmup: int = 3,
    trials: int = 20,
    peak: DevicePeak = REFERENCE_PEAK,
) -> BenchmarkResult:
    """Time ``fn`` to steady state and derive roofline metrics.

    Reports the *best* (minimum) per-trial latency as the estimate, which is the
    standard practice for isolating kernel cost from host-side scheduler jitter.
    """
    for _ in range(max(0, warmup)):
        fn()
    best = float("inf")
    for _ in range(max(1, trials)):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return BenchmarkResult(name=name, latency_s=best, cost=cost, peak=peak)
