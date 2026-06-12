"""Numerical validation harness (PLAN.md §5).

Three pieces, mirroring the plan's validation layers:

* :mod:`longhornai.validation.tolerance` — the per-dtype numerical-tolerance
  policy that defines what "agrees" means.
* :mod:`longhornai.validation.reference` — float64 golden references that define
  the *semantics* of each operator. Kernels are validated against these.
* :mod:`longhornai.validation.differential` — randomized differential testing
  comparing a backend implementation against the golden reference, with a
  shrinking-style minimal-shape search on failure.
"""

from __future__ import annotations

from .bf16 import BF16_DTYPE_NAME, is_bf16_label, to_bf16
from .differential import DiffResult, differential_test
from .equivalence import (
    BackendOutcome,
    EquivalenceReport,
    assert_cross_target_equivalent,
    run_on_backend,
)
from .reference import REFERENCES
from .regression import (
    RegressionFinding,
    RegressionReport,
    run_regression_check,
)
from .readiness import (
    ReadinessCard,
    ReadinessCriterion,
    format_readiness_card,
    run_readiness_card,
)
from .silicon_smoke import (
    SMOKE_FIXTURES,
    SmokeOutcome,
    SmokeReport,
    run_full_cross_target_sweep,
    run_silicon_smoke,
)
from .tolerance import Tolerance, assert_close, tolerance_for

__all__ = [
    "Tolerance",
    "tolerance_for",
    "assert_close",
    "REFERENCES",
    "differential_test",
    "DiffResult",
    "to_bf16",
    "is_bf16_label",
    "BF16_DTYPE_NAME",
    "assert_cross_target_equivalent",
    "EquivalenceReport",
    "BackendOutcome",
    "run_on_backend",
    # M7 silicon bring-up
    "SmokeOutcome",
    "SmokeReport",
    "SMOKE_FIXTURES",
    "run_silicon_smoke",
    "run_full_cross_target_sweep",
    # M8 regression hardening
    "RegressionFinding",
    "RegressionReport",
    "run_regression_check",
    # M8 readiness
    "ReadinessCard",
    "ReadinessCriterion",
    "run_readiness_card",
    "format_readiness_card",
]
