"""Longhorn Silicon backend (PLAN.md §2.2 / §8 M7).

PLAN.md §8 M7 deliverable: "silicon bring-up (first part)". Until first
silicon arrives the lhsil backend is a CPU-equivalent shim — same surface,
same numerics, exercised by the cross-target equivalence harness alongside
CPU / sim / RTL / FPGA. The M7 exit gate ("Silicon smoke + per-kernel
correctness pass; cross-target equivalence holds") runs through this
backend.

When real silicon lands, the per-op delegations switch to the lhsil
runtime driver; the registration surface stays unchanged.
"""

from __future__ import annotations

from ..runtime.backend import register_backend
from ._clone import build_cpu_clone_backend

lhsil = build_cpu_clone_backend(
    "lhsil",
    "Longhorn Silicon backend (CPU-equivalent until first silicon lands).",
)
register_backend(lhsil)
