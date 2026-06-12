"""GPTQ — Second-order weight quantization with per-column residual correction.

PLAN.md §3 Phase 4. GPTQ frames quantization as minimizing
``||X (W - Q) ||²_F`` per output column, with the Hessian
``H = X^T X`` providing the second-order information. Each column is
quantized one row at a time; after each quantization the residual error
is *spread* to the remaining un-quantized rows via ``H^{-1}``, so later
rows compensate for earlier rounding decisions.

This file ships a faithful but small reference impl: groupwise INT4
weights with a Cholesky-of-the-inverse-Hessian-driven residual update.
The output is a packed INT4 weight and groupwise scales — drop-in for
the W4A16 GEMM kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import pack_int4


@dataclass(frozen=True)
class GPTQResult:
    weight_packed: np.ndarray       # int8 (K // 2, N) packed INT4
    scale_groupwise: np.ndarray     # (K // group_size, N) FP32 group scales
    group_size: int


def _round_to_int4(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Symmetric INT4 round with clipping to [-8, 7]."""
    q = np.rint(x.astype(np.float64) / scale)
    return np.clip(q, -8, 7).astype(np.int32)


def gptq_calibrate(
    weight: np.ndarray,
    activations: np.ndarray,
    *,
    group_size: int = 32,
    damping: float = 0.01,
) -> GPTQResult:
    """Run GPTQ calibration for one ``(K, N)`` weight matrix.

    ``weight``: ``(K, N)``. ``activations``: ``(N_calib, K)`` — calibration
    activations stacked along axis 0.
    ``damping``: diagonal load on the Hessian to stabilize the inversion.
    """
    K, N = weight.shape
    if K % group_size != 0:
        raise ValueError(f"GPTQ: K ({K}) must be divisible by group_size ({group_size})")

    # Compute the un-normalized Hessian H = X^T X / N_calib.
    flat_act = activations.reshape(-1, K).astype(np.float64)
    H = (flat_act.T @ flat_act) / max(flat_act.shape[0], 1)
    H[np.arange(K), np.arange(K)] += damping * np.mean(np.diag(H))

    # Cholesky-based inverse-Hessian factor for residual updates.
    try:
        Hinv = np.linalg.inv(H)
        L = np.linalg.cholesky(Hinv).T   # upper-triangular factor
    except np.linalg.LinAlgError:
        # Fall back to identity → degenerates to plain groupwise quant.
        L = np.eye(K)

    W = weight.astype(np.float64).copy()
    n_groups = K // group_size
    scale_g = np.zeros((n_groups, N), dtype=np.float32)
    Q = np.zeros((K, N), dtype=np.int32)

    for g in range(n_groups):
        i0, i1 = g * group_size, (g + 1) * group_size
        W_block = W[i0:i1]                                  # (group_size, N)
        amax = np.maximum(np.abs(W_block).max(axis=0), 1e-12)
        scale = (amax / 7.0).astype(np.float32)             # per-column scale
        scale_g[g] = scale
        for k in range(group_size):
            row = i0 + k
            w_row = W[row].copy()
            q = _round_to_int4(w_row, scale)
            Q[row] = q
            dq = (q.astype(np.float64) * scale)
            err = w_row - dq
            # Spread the rounding residual to remaining rows in the same and
            # following groups via the Cholesky factor — this is the "GPTQ
            # update" with the Hessian-aware compensation.
            if row + 1 < K:
                W[row + 1:] -= np.outer(L[row, row + 1:], err)

    packed = pack_int4(Q.astype(np.int8), axis=0)
    return GPTQResult(weight_packed=packed, scale_groupwise=scale_g, group_size=group_size)


__all__ = ["GPTQResult", "gptq_calibrate"]
