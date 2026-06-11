"""Activation kernels: GELU and SiLU."""

from __future__ import annotations

import numpy as np

from ..runtime import dispatch


def gelu(x: np.ndarray, approximate: str = "none"):
    """GELU activation. ``approximate`` is ``"none"`` (erf) or ``"tanh"``."""
    if approximate not in ("none", "tanh"):
        raise ValueError("gelu approximate must be 'none' or 'tanh'")
    return dispatch("gelu", x, approximate=approximate)


def silu(x: np.ndarray):
    """SiLU / swish activation: x * sigmoid(x)."""
    return dispatch("silu", x)
