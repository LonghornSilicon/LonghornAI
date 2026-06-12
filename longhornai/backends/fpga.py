"""FPGA prototype backend (PLAN.md §2.2 / §8 M4).

PLAN.md §8 M4 deliverable: "first FPGA bring-up". Until the FPGA bitstream
flow is wired up the backend is a CPU-equivalent shim — same surface, same
numerics, same equivalence-harness coverage. M4 exit gate ("E2E Llama
decode under continuous batching on FPGA") runs through this backend.

When a real FPGA prototype lands the per-op delegations switch to the
FPGA runtime driver; the registration surface stays unchanged.
"""

from __future__ import annotations

from ..runtime.backend import register_backend
from ._clone import build_cpu_clone_backend

fpga = build_cpu_clone_backend(
    "fpga",
    "FPGA prototype shim (CPU-equivalent until bitstream flow lands).",
)
register_backend(fpga)
