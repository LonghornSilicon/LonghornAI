"""GEMM shape suite — the Phase-1 exit-gate measurement set.

PLAN.md §3 Phase 1 exit gate: "GEMM within target % of reference peak on
the shape suite". §4.3 says shapes are drawn from real model configs (head
dims, hidden sizes, vocab, seq lengths from the §7 models). This module
defines that suite, runs it through the active backend, and reports the
PLAN.md §4.2 metric set per shape.

The recorded baseline lives in ``benchmarks/gemm_baseline.json``. M1
captures the CPU NumPy reference baseline; production targets land on
silicon (PLAN.md §9.1, §M8).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..kernels import gemm
from .cost import cost_gemm
from .harness import REFERENCE_PEAK, BenchmarkResult, DevicePeak, benchmark


@dataclass(frozen=True)
class ShapeCase:
    """A single GEMM shape, named by the model layer it represents."""

    name: str
    M: int
    K: int
    N: int


# Shapes drawn from PLAN.md §7 model configs. Compute-bound (square + prefill)
# and memory-bound (decode skinny) cases both appear so the suite exercises
# both rooflines. Comments cite the model whose config motivated the entry.
GEMM_SHAPE_SUITE: List[ShapeCase] = [
    # Canonical square benchmarks — direct compare to cuBLAS shape tables.
    ShapeCase("square_1k", 1024, 1024, 1024),
    ShapeCase("square_2k", 2048, 2048, 2048),
    # Llama-7B / Mistral-7B prefill (hidden=4096, intermediate=11008).
    ShapeCase("llama7b_qkv_proj", 4096, 4096, 4096 * 3),
    ShapeCase("llama7b_mlp_gate_up", 4096, 4096, 11008),
    ShapeCase("llama7b_mlp_down", 4096, 11008, 4096),
    # Llama-3-8B (hidden=4096, GQA, intermediate=14336, vocab=128k).
    ShapeCase("llama3_8b_mlp_down", 4096, 14336, 4096),
    ShapeCase("llama3_8b_lm_head", 4096, 4096, 128_256),
    # Qwen-2.5 / Mistral-7B attention output projection (head_dim=128, n_heads=32).
    ShapeCase("qwen_attn_o_proj", 4096, 4096, 4096),
    # Decode-time skinny GEMMs — memory-bound, dominated by weight bandwidth.
    ShapeCase("decode_proj_4k", 1, 4096, 4096),
    ShapeCase("decode_lm_head_8b", 1, 4096, 128_256),
    ShapeCase("decode_mlp_down_7b", 1, 11008, 4096),
]


@dataclass(frozen=True)
class ShapeResult:
    """Per-shape measurement and roofline classification."""

    name: str
    M: int
    K: int
    N: int
    dtype: str
    latency_ms: float
    achieved_tflops: float
    achieved_gbps: float
    arith_intensity: float
    is_compute_bound: bool
    flops_utilization: float
    bandwidth_utilization: float
    roofline_efficiency: float

    @classmethod
    def from_benchmark(cls, case: ShapeCase, dtype: str, r: BenchmarkResult) -> "ShapeResult":
        return cls(
            name=case.name,
            M=case.M,
            K=case.K,
            N=case.N,
            dtype=dtype,
            latency_ms=r.latency_s * 1e3,
            achieved_tflops=r.achieved_flops / 1e12,
            achieved_gbps=r.achieved_bandwidth / 1e9,
            arith_intensity=r.cost.arithmetic_intensity,
            is_compute_bound=r.is_compute_bound,
            flops_utilization=r.flops_utilization,
            bandwidth_utilization=r.bandwidth_utilization,
            roofline_efficiency=r.roofline_efficiency,
        )


def run_gemm_suite(
    *,
    dtype=np.float16,
    cases: Optional[List[ShapeCase]] = None,
    peak: DevicePeak = REFERENCE_PEAK,
    warmup: int = 2,
    trials: int = 5,
    seed: int = 0,
) -> List[ShapeResult]:
    """Run the GEMM shape suite and return per-shape metrics."""
    cases = list(cases) if cases is not None else list(GEMM_SHAPE_SUITE)
    rng = np.random.default_rng(seed)
    out: List[ShapeResult] = []
    dt_name = np.dtype(dtype).name
    for c in cases:
        a = rng.standard_normal((c.M, c.K)).astype(dtype)
        b = rng.standard_normal((c.K, c.N)).astype(dtype)
        cost = cost_gemm(c.M, c.K, c.N, dtype=dtype)
        r = benchmark(
            lambda a=a, b=b: gemm(a, b),
            cost,
            name=f"gemm[{c.M}x{c.K}x{c.N}] {dt_name}",
            warmup=warmup,
            trials=trials,
            peak=peak,
        )
        out.append(ShapeResult.from_benchmark(c, dt_name, r))
    return out


def format_suite_report(results: List[ShapeResult]) -> str:
    """Human-readable table of per-shape metrics."""
    header = (
        f"{'shape':<26} {'dtype':<8} {'lat (ms)':>10} "
        f"{'TFLOP/s':>9} {'GB/s':>9} {'AI':>7} {'bound':>7} {'roof %':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        bound = "compute" if r.is_compute_bound else "memory"
        lines.append(
            f"{r.name:<26} {r.dtype:<8} {r.latency_ms:>10.3f} "
            f"{r.achieved_tflops:>9.3f} {r.achieved_gbps:>9.2f} "
            f"{r.arith_intensity:>7.1f} {bound:>7} "
            f"{r.roofline_efficiency * 100:>7.2f}%"
        )
    return "\n".join(lines)


_BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "gemm_baseline.json"


def baseline_path() -> pathlib.Path:
    return _BASELINE_PATH


def load_baseline() -> Dict[str, Any]:
    if not _BASELINE_PATH.exists():
        return {"schema": 1, "shapes": []}
    return json.loads(_BASELINE_PATH.read_text())


def save_baseline(results: List[ShapeResult], *, note: str = "") -> None:
    """Persist suite measurements as the new baseline."""
    payload = {
        "schema": 1,
        "note": note
        or "CPU NumPy reference baseline; production targets land on silicon (PLAN.md §9.1).",
        "shapes": [asdict(r) for r in results],
    }
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2))


def compare_to_baseline(
    results: List[ShapeResult],
    baseline: Dict[str, Any],
    *,
    regression_factor: float = 1.5,
) -> List[Dict[str, Any]]:
    """Compare results to a baseline snapshot; flag latency regressions.

    A shape regresses when its measured latency exceeds the baseline by
    ``regression_factor`` × (default 1.5×). Shapes absent from the baseline
    are reported with ``baseline_ms = None`` and ``regressed = False``.
    """
    by_name = {s["name"]: s for s in baseline.get("shapes", [])}
    rows: List[Dict[str, Any]] = []
    for r in results:
        prev = by_name.get(r.name)
        prev_lat = prev["latency_ms"] if prev else None
        regressed = bool(prev_lat is not None and r.latency_ms > prev_lat * regression_factor)
        rows.append(
            {
                "name": r.name,
                "latency_ms": r.latency_ms,
                "baseline_ms": prev_lat,
                "regressed": regressed,
                "ratio": (r.latency_ms / prev_lat) if prev_lat else None,
            }
        )
    return rows


__all__ = [
    "ShapeCase",
    "ShapeResult",
    "GEMM_SHAPE_SUITE",
    "run_gemm_suite",
    "format_suite_report",
    "load_baseline",
    "save_baseline",
    "compare_to_baseline",
    "baseline_path",
]
