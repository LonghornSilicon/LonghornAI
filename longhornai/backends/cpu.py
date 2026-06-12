"""CPU reference backend (NumPy) — orchestrator.

The portable, always-available target. It is the golden-generation and CI
default (PLAN.md §2.2) and the correctness anchor every other backend is
validated against via the cross-target equivalence harness.

Per PLAN.md §2.1 each kernel family carries its own backend impl in
``kernels/<family>/impl.py``. This module constructs the singleton CPU
:class:`Backend`, exposes the ``_acc`` accumulation-dtype helper that every
family impl uses, and triggers the family-impl imports so they self-register
on import. Doing the registrations here — rather than at the family
``__init__.py`` — keeps front-end imports cheap and confines the
backend-specific dependency direction to one place.
"""

from __future__ import annotations

import numpy as np

from ..runtime.backend import Backend, register_backend

cpu = Backend("cpu", "Portable NumPy reference backend (golden / CI default).")


def _acc(dtype: np.dtype) -> np.dtype:
    """Accumulation dtype for the reference anchor: float64 for any float input.

    Implementation policy (PLAN.md §3 Phase 1): the CPU backend is the
    correctness *anchor*, so it accumulates in the widest practical precision
    (float64) and rounds only the final result back to the I/O dtype. The
    FP16/BF16-in, FP32-accum contract of real tensor cores is modeled by the
    hardware-representative backends (sim / lhsil), where accumulation
    precision is part of what we are validating — not here.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return np.dtype(np.float64)
    return dt


# Trigger per-family CPU-impl registrations. Each module decorates `cpu`. Order
# is irrelevant since registrations are independent, but we group them by
# family for readability.
from ..kernels.gemm import impl as _gemm_impl  # noqa: F401, E402
from ..kernels.normalization import impl as _norm_impl  # noqa: F401, E402
from ..kernels.activation import impl as _act_impl  # noqa: F401, E402
from ..kernels.elementwise import impl as _elem_impl  # noqa: F401, E402
from ..kernels.attention import impl as _attn_impl  # noqa: F401, E402
from ..kernels.kv_cache import impl as _kv_impl  # noqa: F401, E402
from ..kernels.paged_attention import impl as _paged_impl  # noqa: F401, E402
from ..kernels.quant import impl as _quant_impl  # noqa: F401, E402
from ..kernels.collectives import impl as _collectives_impl  # noqa: F401, E402
from ..kernels.moe import impl as _moe_impl  # noqa: F401, E402
from ..kernels.sparse_attention import impl as _sparse_impl  # noqa: F401, E402
from ..kernels.fused import impl as _fused_impl  # noqa: F401, E402

register_backend(cpu, default=True)
