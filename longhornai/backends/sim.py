"""Architectural simulator backend (PLAN.md §2.2).

Cycle-approximate target used for performance modelling — occupancy and
cycle estimates per kernel before silicon. Until the simulator lands as a
proper cycle model the backend is a CPU-equivalent shim, mirroring the
RTL/FPGA approach: same surface, same numerics, exercised by the same
cross-target equivalence harness.
"""

from __future__ import annotations

from ..runtime.backend import register_backend
from ._clone import build_cpu_clone_backend

sim = build_cpu_clone_backend(
    "sim",
    "Architectural simulator shim (CPU-equivalent until cycle model lands).",
)
register_backend(sim)
