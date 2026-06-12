"""SmoothQuant — activation-outlier migration enabling W8A8 accuracy.

PLAN.md §3 Phase 4. Activations of LLMs have a few extreme channels
(outliers); per-tensor INT8 quantization clips them and accuracy collapses.
SmoothQuant rebalances by moving the difficulty from activations into
weights:

    s_c = max(|X[:, c]|)^alpha / max(|W[c, :]|)^(1 - alpha)
    X' = X / s          (cheaper to quantize — outliers tamed)
    W' = W * s          (slightly harder, but weights are static + per-channel)

After the migration both X' and W' have well-behaved ranges, so a plain
INT8 quant on each preserves accuracy.

The fold ``X' = X / s`` is absorbed into the previous layer's output
projection at deployment; here we return the scale and the migrated
weight so the test path can verify the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SmoothQuantResult:
    """Output of SmoothQuant calibration."""

    scale: np.ndarray             # (K,) per-input-channel migration scale
    weight_smoothed: np.ndarray   # (K, N) — W * s (broadcast over output dim)
    alpha: float                  # the migration strength used


def smooth_quant_calibrate(
    weight: np.ndarray,
    activations: np.ndarray,
    *,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> SmoothQuantResult:
    """Compute the SmoothQuant migration scale and the migrated weight.

    ``weight``: ``(K, N)`` — the layer's weight (K is the input/reduction dim).
    ``activations``: ``(..., K)`` — calibration activations, leading dims
        are batch/sequence and the last is the channel dim that matches ``K``.
    ``alpha``: 0..1 — migration strength. 0 keeps activations untouched
        (reduces to vanilla weight quant); 1 dumps all of the dynamic range
        onto weights. 0.5 is the SmoothQuant paper default.
    """
    if weight.shape[0] != activations.shape[-1]:
        raise ValueError(
            f"SmoothQuant: weight K={weight.shape[0]} must match "
            f"activations channel dim {activations.shape[-1]}"
        )
    act_max = np.maximum(
        np.max(np.abs(activations.reshape(-1, activations.shape[-1])), axis=0),
        eps,
    )
    weight_max = np.maximum(np.max(np.abs(weight), axis=1), eps)
    scale = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
    scale = np.maximum(scale, eps).astype(np.float32)
    weight_smoothed = (weight.astype(np.float64) * scale[:, None]).astype(weight.dtype)
    return SmoothQuantResult(scale=scale, weight_smoothed=weight_smoothed, alpha=alpha)


def smooth_quant_apply(
    activation: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    """Apply ``X / s`` to an activation post-calibration. Last-axis broadcast."""
    return (activation.astype(np.float64) / scale).astype(activation.dtype)


__all__ = ["SmoothQuantResult", "smooth_quant_calibrate", "smooth_quant_apply"]
