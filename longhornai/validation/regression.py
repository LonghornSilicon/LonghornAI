"""Regression-hardening harness (PLAN.md §4.3 / §8 M8).

PLAN.md §4.3 calls for "merges that regress a contract beyond threshold
are blocked". This module is the gate: it drives every checked-in baseline
(GEMM shape suite, decode tokens/sec, full performance sweep), compares to
the recorded baseline, and aggregates the result into a single pass/fail
report that CI can consume.

The four baselines covered:

* ``benchmarks/gemm_baseline.json`` — per-shape GEMM latency.
* ``benchmarks/decode_baseline.json`` — continuous-batching tokens/sec.
* ``benchmarks/full_sweep_baseline.json`` — every M8 perf row.
* Cross-target equivalence — numerical, not perf, but blocking too.

CLI: ``lhai regression-check`` runs the gate and exits non-zero on
regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..benchmarks import (
    DecodeBenchmarkResult,
    compare_to_baseline as compare_gemm,
    load_baseline as load_gemm_baseline,
    load_decode_baseline,
    run_decode_benchmark,
    run_gemm_suite,
)
from ..benchmarks.full_sweep import (
    PerfRow,
    compare_full_sweep,
    load_full_sweep_baseline,
    run_full_sweep,
)
from ..benchmarks.shape_suite import GEMM_SHAPE_SUITE
from ..models import LlamaConfig, init_random_weights
from .silicon_smoke import run_full_cross_target_sweep


@dataclass
class RegressionFinding:
    """One regressed measurement."""

    suite: str
    name: str
    baseline_ms: float | None
    measured_ms: float
    ratio: float | None


@dataclass
class RegressionReport:
    """Aggregate report covering all baselines."""

    findings: List[RegressionFinding] = field(default_factory=list)
    cross_target_failures: List[str] = field(default_factory=list)
    suites_skipped: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings and not self.cross_target_failures

    def summary(self) -> str:
        if self.passed:
            head = f"OK — no regressions across {3 - len(self.suites_skipped)} suites"
        else:
            head = (
                f"FAIL — {len(self.findings)} latency regression(s) + "
                f"{len(self.cross_target_failures)} cross-target failure(s)"
            )
        lines = [head]
        for f in self.findings:
            lines.append(
                f"  [{f.suite}] {f.name}: {f.measured_ms:.3f} ms vs "
                f"baseline {f.baseline_ms:.3f} ms "
                f"({f.ratio:.2f}× over budget)"
            )
        for op in self.cross_target_failures:
            lines.append(f"  [cross-target] {op} disagreed across backends")
        for s in self.suites_skipped:
            lines.append(f"  [skipped] {s}: no baseline recorded yet")
        return "\n".join(lines)


def _gemm_quick_subset():
    wanted = {"square_1k", "decode_proj_4k", "qwen_attn_o_proj"}
    return [c for c in GEMM_SHAPE_SUITE if c.name in wanted]


def run_regression_check(
    *, regression_factor: float = 1.5, dtype=np.float32,
) -> RegressionReport:
    """Run every baseline check and emit a single aggregated report."""
    report = RegressionReport()

    # 1) GEMM shape suite.
    base = load_gemm_baseline()
    if base.get("shapes"):
        results = run_gemm_suite(
            dtype=dtype, cases=_gemm_quick_subset(), warmup=1, trials=3,
        )
        rows = compare_gemm(results, base, regression_factor=regression_factor)
        for r in rows:
            if r["regressed"]:
                report.findings.append(RegressionFinding(
                    suite="gemm", name=r["name"],
                    baseline_ms=r["baseline_ms"], measured_ms=r["latency_ms"],
                    ratio=r["ratio"],
                ))
    else:
        report.suites_skipped.append("gemm")

    # 2) Decode tokens/sec.
    decode_base = load_decode_baseline()
    if decode_base.get("results"):
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
        )
        weights = init_random_weights(config, dtype=dtype, seed=0)
        result = run_decode_benchmark(
            config=config, weights=weights,
            num_requests=4, prompt_len=8, max_new_tokens=8,
            label="regression-check",
        )
        # Compare aggregate tokens/sec — a *floor* (must be ≥ 1/factor of baseline).
        recorded = decode_base["results"][0]
        floor = recorded["tokens_per_second"] / regression_factor
        if result.tokens_per_second < floor:
            report.findings.append(RegressionFinding(
                suite="decode", name=recorded.get("label", "decode"),
                baseline_ms=recorded["wall_time_s"] * 1e3,
                measured_ms=result.wall_time_s * 1e3,
                ratio=(result.wall_time_s
                        / recorded["wall_time_s"]) if recorded["wall_time_s"] else None,
            ))
    else:
        report.suites_skipped.append("decode")

    # 3) Full performance sweep.
    full_base = load_full_sweep_baseline()
    if full_base.get("rows"):
        rows = run_full_sweep(dtype=dtype)
        cmp = compare_full_sweep(rows, full_base, regression_factor=regression_factor)
        for c in cmp:
            if c["regressed"]:
                report.findings.append(RegressionFinding(
                    suite="full_sweep",
                    name=f"{c['family']}/{c['op']}/{c['shape']}",
                    baseline_ms=c["baseline_ms"], measured_ms=c["latency_ms"],
                    ratio=c["ratio"],
                ))
    else:
        report.suites_skipped.append("full_sweep")

    # 4) Cross-target equivalence — numerical contract, not perf.
    sweep = run_full_cross_target_sweep()
    for op, eq in sweep.items():
        if not eq.passed:
            report.cross_target_failures.append(op)

    return report


__all__ = [
    "RegressionFinding",
    "RegressionReport",
    "run_regression_check",
]
