"""Activation kernels: GELU and SiLU.

Family layout per PLAN.md §2.1: ``__init__.py`` (front-ends), ``reference.py``
(float64 goldens), ``impl.py`` (CPU impls), ``tuning.py`` (search spaces),
and ``KERNEL.md`` (contract).
"""

from __future__ import annotations

import numpy as np

from ...runtime import dispatch


def gelu(x: np.ndarray, approximate: str = "none"):
    """GELU activation. ``approximate`` is ``"none"`` (erf) or ``"tanh"``."""
    if approximate not in ("none", "tanh"):
        raise ValueError("gelu approximate must be 'none' or 'tanh'")
    return dispatch("gelu", x, approximate=approximate)


def silu(x: np.ndarray):
    """SiLU / swish activation: x * sigmoid(x)."""
    return dispatch("silu", x)


__all__ = ["gelu", "silu"]
