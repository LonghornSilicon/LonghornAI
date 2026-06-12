"""RTL co-simulation backend (PLAN.md §2.2 / §8 M3).

The RTL backend is a co-simulation shim until Verilator/SystemVerilog hooks
land alongside actual RTL. Every op delegates to the CPU reference; the
backend is registered, selectable, and exercised by the cross-target
equivalence harness. PLAN.md §8 M3 exit gate: **RTL ≡ CPU on attention** —
enforced by :mod:`longhornai.validation.equivalence`.

When real RTL co-simulation lands the per-op delegations switch to Verilator
IPC; the registration surface and the equivalence harness stay unchanged.
"""

from __future__ import annotations

from ..runtime.backend import register_backend
from ._clone import build_cpu_clone_backend

rtl = build_cpu_clone_backend(
    "rtl",
    "RTL co-simulation shim (CPU-equivalent until Verilator hookups land).",
)
register_backend(rtl)
