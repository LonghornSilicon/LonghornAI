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

from .differential import DiffResult, differential_test
from .reference import REFERENCES
from .tolerance import Tolerance, assert_close, tolerance_for

__all__ = [
    "Tolerance",
    "tolerance_for",
    "assert_close",
    "REFERENCES",
    "differential_test",
    "DiffResult",
]
