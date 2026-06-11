"""Normalization kernels: LayerNorm and RMSNorm."""

from __future__ import annotations

import numpy as np

from ..runtime import dispatch


def layernorm(x: np.ndarray, weight=None, bias=None, eps: float = 1e-5, axis: int = -1):
    """LayerNorm over ``axis`` with optional affine weight/bias."""
    return dispatch("layernorm", x, weight=weight, bias=bias, eps=eps, axis=axis)


def rmsnorm(x: np.ndarray, weight=None, eps: float = 1e-6, axis: int = -1):
    """RMSNorm over ``axis`` (Llama/Qwen/Mistral norm) with optional weight."""
    return dispatch("rmsnorm", x, weight=weight, eps=eps, axis=axis)
