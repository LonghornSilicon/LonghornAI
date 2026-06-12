"""Production-inference readiness scorecard (PLAN.md §8 M8 exit gate).

The M8 exit gate is "Production inference readiness criteria met (§9)".
PLAN.md §9 enumerates kernel KPIs (GEMM % of cuBLAS, FA parity, roofline
efficiency, quant accuracy), system KPIs (tokens/sec, latency SLO, scaling
efficiency), and program KPIs (silicon bring-up readiness, cross-target
equivalence, production inference readiness, coverage + regression health).

This module is the *aggregator*: it walks every readiness check that's
testable in this repo and emits a pass/fail line per criterion. CI wires
``lhai readiness`` to gate releases.

The criteria covered map directly to PLAN.md §9.3 KPIs:

* **Silicon bring-up readiness** — every smoke fixture passes on lhsil.
* **Cross-target equivalence** — every certified op agrees across the 5
  pre-silicon backends.
* **Production inference readiness** — :class:`LonghornInference`
  generates correct tokens end-to-end on every backend.
* **Coverage & regression health** — pytest collection + checked-in
  baselines round-trip cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

import numpy as np

from ..models import LlamaConfig, init_random_weights
from ..runtime import LonghornInference, available_backends, get_backend
from .silicon_smoke import run_full_cross_target_sweep, run_silicon_smoke


@dataclass
class ReadinessCriterion:
    """One row in the readiness scorecard."""

    name: str
    passed: bool
    detail: str


@dataclass
class ReadinessCard:
    """Aggregate scorecard for the M8 exit gate."""

    criteria: List[ReadinessCriterion] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def num_pass(self) -> int:
        return sum(1 for c in self.criteria if c.passed)


# --- individual criteria ---------------------------------------------------

def _criterion_silicon_bringup_smoke() -> ReadinessCriterion:
    """PLAN.md §9.3: bring-up validation suite passing on lhsil."""
    report = run_silicon_smoke(backend="lhsil", reference="cpu")
    return ReadinessCriterion(
        name="silicon bring-up smoke (lhsil)",
        passed=report.passed,
        detail=f"{len(report.outcomes)} ops; {report.num_failed} failure(s)",
    )


def _criterion_cross_target_equivalence() -> ReadinessCriterion:
    """PLAN.md §9.3: every certified op agrees across 5 pre-silicon backends."""
    sweep = run_full_cross_target_sweep()
    failed = [op for op, eq in sweep.items() if not eq.passed]
    backends = available_backends()
    return ReadinessCriterion(
        name="cross-target equivalence (5 backends)",
        passed=not failed,
        detail=f"{len(sweep)} ops × {len(backends)} backends; "
               f"failed: {failed if failed else 'none'}",
    )


def _criterion_inference_end_to_end() -> ReadinessCriterion:
    """LonghornInference produces a finite, shape-correct generation."""
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    try:
        with LonghornInference(weights, config) as inf:
            results = inf.generate(
                [[1, 2, 3, 4], [5, 6, 7]], max_new_tokens=3,
            )
        # Every prompt must produce exactly max_new_tokens output IDs.
        ok = all(len(r.output_ids) == 3 for r in results)
        return ReadinessCriterion(
            name="LonghornInference end-to-end",
            passed=ok,
            detail=f"{len(results)} request(s); "
                   f"output lens: {[len(r.output_ids) for r in results]}",
        )
    except Exception as exc:  # pragma: no cover - failure path
        return ReadinessCriterion(
            name="LonghornInference end-to-end",
            passed=False, detail=repr(exc),
        )


def _criterion_inference_under_each_backend() -> ReadinessCriterion:
    """The same inference call produces matching outputs on every backend."""
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8,
    )
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = [[1, 2, 3]]
    refs: List[Tuple[str, List[int]]] = []
    for backend in available_backends():
        with LonghornInference(weights, config) as inf:
            r = inf.generate(prompt, max_new_tokens=2, backend=backend)
        refs.append((backend, list(r[0].output_ids)))
    cpu_out = next(out for b, out in refs if b == "cpu")
    diverged = [b for b, out in refs if out != cpu_out]
    return ReadinessCriterion(
        name="inference parity across backends",
        passed=not diverged,
        detail=f"backends: {[b for b, _ in refs]}; "
               f"diverged: {diverged if diverged else 'none'}",
    )


def _criterion_baselines_present() -> ReadinessCriterion:
    """Coverage: every checked-in baseline JSON parses cleanly."""
    from ..benchmarks.decode_bench import baseline_path as decode_path
    from ..benchmarks.full_sweep import baseline_path as full_path
    from ..benchmarks.shape_suite import baseline_path as gemm_path
    import json
    paths = [gemm_path(), decode_path(), full_path()]
    valid: List[str] = []
    invalid: List[str] = []
    for p in paths:
        if p.exists():
            try:
                payload = json.loads(p.read_text())
                if "schema" in payload:
                    valid.append(p.name)
                else:
                    invalid.append(p.name)
            except Exception as exc:
                invalid.append(f"{p.name} ({exc})")
        else:
            invalid.append(f"{p.name} (missing)")
    return ReadinessCriterion(
        name="baseline JSON files round-trip",
        passed=not invalid,
        detail=f"valid: {valid}; invalid: {invalid if invalid else 'none'}",
    )


def _criterion_5_backends_registered() -> ReadinessCriterion:
    """All five pre-silicon backends register the full op surface."""
    expected = {"cpu", "sim", "rtl", "fpga", "lhsil"}
    actual = set(available_backends())
    cpu_ops = set(get_backend("cpu").ops())
    extras = []
    for name in expected & actual:
        ops = set(get_backend(name).ops())
        if ops != cpu_ops:
            extras.append(f"{name}: {len(ops)} vs cpu {len(cpu_ops)}")
    ok = expected.issubset(actual) and not extras
    return ReadinessCriterion(
        name="5 backends registered with full op surface",
        passed=ok,
        detail=f"present: {sorted(actual)}; mismatch: {extras}",
    )


# --- top-level entry --------------------------------------------------------

def run_readiness_card() -> ReadinessCard:
    """Run every readiness check and aggregate."""
    card = ReadinessCard()
    card.criteria.extend([
        _criterion_5_backends_registered(),
        _criterion_silicon_bringup_smoke(),
        _criterion_cross_target_equivalence(),
        _criterion_inference_end_to_end(),
        _criterion_inference_under_each_backend(),
        _criterion_baselines_present(),
    ])
    return card


def format_readiness_card(card: ReadinessCard) -> str:
    head = f"Production-inference readiness — PLAN.md §9 / M8 exit gate"
    div = "-" * len(head)
    lines = [head, div]
    for c in card.criteria:
        tag = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{tag}] {c.name}")
        lines.append(f"         {c.detail}")
    summary = (
        f"\n{card.num_pass}/{len(card.criteria)} criteria passed; "
        f"overall: {'OK' if card.passed else 'FAIL'}"
    )
    lines.append(summary)
    return "\n".join(lines)


__all__ = [
    "ReadinessCriterion",
    "ReadinessCard",
    "run_readiness_card",
    "format_readiness_card",
]
