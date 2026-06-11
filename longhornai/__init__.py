"""LonghornAI — canonical AI inference kernel library for Longhorn Silicon.

This package is the M1 reference implementation: Phase-1 foundational compute
kernels backed by a portable CPU (NumPy) reference backend, a backend-dispatch
runtime, a numerical validation harness, and a benchmarking harness.

The public surface is intentionally small and stable:

    >>> import longhornai as lh
    >>> import numpy as np
    >>> a = np.random.randn(64, 128).astype("float32")
    >>> b = np.random.randn(128, 32).astype("float32")
    >>> c = lh.gemm(a, b)            # dispatches to the active backend
    >>> c.shape
    (64, 32)

Kernels are dispatched through :mod:`longhornai.runtime`, so the same call
re-targets across backends (CPU today; sim / RTL / FPGA / silicon per PLAN.md)
without changing user code.
"""

from __future__ import annotations

from .runtime import (
    Backend,
    available_backends,
    dispatch,
    get_backend,
    set_default_backend,
    use_backend,
)
from .kernels import (
    batched_gemm,
    embedding_lookup,
    gelu,
    gemm,
    grouped_gemm,
    layernorm,
    reduce,
    rmsnorm,
    rope,
    silu,
    softmax,
    tensor_contraction,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # runtime
    "Backend",
    "dispatch",
    "get_backend",
    "set_default_backend",
    "available_backends",
    "use_backend",
    # kernels (Phase 1)
    "gemm",
    "batched_gemm",
    "grouped_gemm",
    "tensor_contraction",
    "layernorm",
    "rmsnorm",
    "softmax",
    "gelu",
    "silu",
    "rope",
    "embedding_lookup",
    "reduce",
]
