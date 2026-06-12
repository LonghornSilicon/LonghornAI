"""Helpers for building backends that delegate to a reference implementation.

Many pre-silicon backends (sim, rtl, fpga) start their lives as
*CPU-equivalent shims* — the registration surface, dispatch wiring, and
cross-target equivalence harness exist before the real implementation lands.
This module factors the "register every CPU op as a delegate" pattern so each
shim is one short module.

When a real implementation arrives, the per-op delegate becomes the call
into Verilator / FPGA-runtime / lhsil-driver and the rest of the system
stays unchanged.
"""

from __future__ import annotations

from ..runtime.backend import Backend
from . import cpu as _cpu


def build_cpu_clone_backend(name: str, description: str) -> Backend:
    """Construct a Backend whose every op delegates to the CPU reference.

    The CPU backend is the correctness anchor (PLAN.md §2.2); cloning its
    op surface gives sim/rtl/fpga shims a behaviorally-equivalent target
    that the cross-target equivalence harness can exercise unchanged.
    """
    backend = Backend(name, description)
    for op_name in _cpu.cpu.ops():
        cpu_impl = _cpu.cpu.get(op_name)

        def _delegate(*args, _impl=cpu_impl, **kwargs):
            return _impl(*args, **kwargs)

        backend.register(op_name)(_delegate)
    return backend


__all__ = ["build_cpu_clone_backend"]
