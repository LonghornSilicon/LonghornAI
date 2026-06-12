"""Normalization kernels: LayerNorm and RMSNorm.

The family is laid out per PLAN.md §2.1:

* :mod:`.reference` — float64 golden references defining numerical semantics
* :mod:`.impl`      — backend-specialized implementations (CPU today)
* :mod:`.tuning`    — auto-tuning configuration spaces
* :file:`KERNEL.md` — input/layout/dtype/perf-target contract
"""

from __future__ import annotations

import numpy as np

from ...runtime import dispatch


def layernorm(x: np.ndarray, weight=None, bias=None, eps: float = 1e-5, axis: int = -1):
    """LayerNorm over ``axis`` with optional affine weight/bias."""
    return dispatch("layernorm", x, weight=weight, bias=bias, eps=eps, axis=axis)


def rmsnorm(x: np.ndarray, weight=None, eps: float = 1e-6, axis: int = -1):
    """RMSNorm over ``axis`` (Llama/Qwen/Mistral norm) with optional weight."""
    return dispatch("rmsnorm", x, weight=weight, eps=eps, axis=axis)


__all__ = ["layernorm", "rmsnorm"]
