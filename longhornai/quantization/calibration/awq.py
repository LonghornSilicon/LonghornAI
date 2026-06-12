"""AWQ — Activation-Aware Weight Quantization.

PLAN.md §3 Phase 4. AWQ observes that a small fraction of weight channels
(those multiplied by high-magnitude activation channels) carry most of the
output information. Pre-multiplying these channels by a per-channel scale
``s`` shifts representation precision toward where it matters; after the
groupwise INT4 quantize the *informative* channels survive at higher
fidelity than naïve quantization would give them.

The deployment fold is ``out = (x / s) @ Q(W * s) ≈ x @ W`` — the ``/s``
is folded into the previous layer's output projection. Here we return the
calibration scale, the calibrated weight, and the resulting INT4 packed
form + groupwise scales.

The reference impl uses a small grid search over ``alpha`` and picks the
value that minimizes ``|x @ W - x @ Q(W * s) / s|`` on the calibration
activations — that's the one-line summary of the AWQ search loop, faithful
to the paper's algorithm 2 at toy scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..primitives import pack_int4, quantize_groupwise


@dataclass(frozen=True)
class AWQResult:
    awq_scale: np.ndarray           # (K,) per-input-channel salience scale
    weight_packed: np.ndarray       # int8 (K // 2, N) packed INT4 of (W * s)
    scale_groupwise: np.ndarray     # (K // group_size, N) FP32 group scales
    group_size: int
    alpha: float                    # chosen by grid search


def _quant_dequant_groupwise_int4(
    weight: np.ndarray, group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper: groupwise-quantize INT4 then dequantize back to FP."""
    q, params = quantize_groupwise(
        weight, bits=4, group_size=group_size, axis=0, asymmetric=False,
    )
    n_groups = weight.shape[0] // group_size
    grouped = q.reshape(n_groups, group_size, -1).astype(np.float64)
    scale_g = params.scale[:, None, :].astype(np.float64)
    dq = (grouped * scale_g).reshape(weight.shape).astype(np.float32)
    return q, params.scale.astype(np.float32), dq


def awq_calibrate(
    weight: np.ndarray,
    activations: np.ndarray,
    *,
    group_size: int = 32,
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> AWQResult:
    """Run AWQ calibration over ``alphas`` and return the best result.

    ``weight``: ``(K, N)``. ``activations``: ``(..., K)`` calibration data.
    """
    if weight.shape[0] != activations.shape[-1]:
        raise ValueError(
            f"AWQ: weight K={weight.shape[0]} != activations channel dim "
            f"{activations.shape[-1]}"
        )
    if weight.shape[0] % group_size != 0:
        raise ValueError(
            f"AWQ: K ({weight.shape[0]}) must be divisible by group_size ({group_size})"
        )

    flat_act = activations.reshape(-1, activations.shape[-1]).astype(np.float64)
    act_mean = np.maximum(np.mean(np.abs(flat_act), axis=0), 1e-8)
    weight_mean = np.maximum(np.mean(np.abs(weight), axis=1), 1e-8)
    weight_f64 = weight.astype(np.float64)
    target = flat_act @ weight_f64

    best = None
    best_err = np.inf
    for alpha in alphas:
        scale = (act_mean ** alpha) / (weight_mean ** (1.0 - alpha))
        scale = np.maximum(scale, 1e-8).astype(np.float32)
        # Calibrated weight Q(W * s) and the dequantized form.
        w_scaled = (weight_f64 * scale[:, None]).astype(np.float32)
        q, scale_g, w_dq = _quant_dequant_groupwise_int4(w_scaled, group_size)
        # Effective forward: (x / s) @ w_dq.  Compare to x @ W (target).
        x_smooth = (flat_act / scale).astype(np.float64)
        approx = x_smooth @ w_dq.astype(np.float64)
        err = float(np.linalg.norm(approx - target))
        if err < best_err:
            best_err = err
            packed = pack_int4(q.astype(np.int8), axis=0)
            best = AWQResult(
                awq_scale=scale,
                weight_packed=packed,
                scale_groupwise=scale_g,
                group_size=group_size,
                alpha=float(alpha),
            )
    assert best is not None
    return best


__all__ = ["AWQResult", "awq_calibrate"]
