"""Fused kernel family (PLAN.md §3 Phase 6 / §8 M7).

The single highest-leverage decode optimization on real silicon — collapse
memory-bound op chains so each operand makes one HBM round-trip. M7 ships
the three fusions that touch every decode step:

* ``rmsnorm_qkv``       — RMSNorm + Q/K/V projections (the prologue of every
  attention block).
* ``attention_output_proj`` — SDPA + output projection (the attention
  epilogue).
* ``gated_mlp``         — SwiGLU / GeGLU MLP fused: gate_proj + up_proj +
  activation·multiply + down_proj.

Each kernel is *numerically equivalent* to its un-fused composition — the
fusion is a memory-traffic optimization, not a math change. The references
encode that equivalence so the differential harness validates fused
backends against the standard chain.

All three are M7 deliverables; the general cross-family fusion framework
("`kernels/fused/` epilogue/prologue", PLAN.md §3 Phase 6) is realized
through these three specific fusions plus the registration scaffolding
(reference / impl / tuning / KERNEL.md).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...runtime import dispatch


def rmsnorm_qkv(
    x: np.ndarray, *,
    norm_weight: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMSNorm(x) -> Q/K/V projections, fused.

    ``x``: ``(tokens, hidden)``.
    Returns ``(q, k, v)`` of the projected dims.
    """
    return dispatch(
        "rmsnorm_qkv", x,
        norm_weight=norm_weight,
        q_weight=q_weight, k_weight=k_weight, v_weight=v_weight,
        eps=eps,
    )


def attention_output_proj(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    o_weight: np.ndarray,
    causal: bool = False, scale: float | None = None,
) -> np.ndarray:
    """SDPA -> output projection, fused.

    ``q``/``k``/``v``: ``(B, n_heads, S, D)``. ``o_weight`` projects the
    concatenated heads back to hidden. Returns ``(B, S, hidden)``.
    """
    return dispatch(
        "attention_output_proj", q, k, v,
        o_weight=o_weight, causal=causal, scale=scale,
    )


def gated_mlp(
    x: np.ndarray, *,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    activation: str = "silu",
) -> np.ndarray:
    """Gated MLP fused: ``activation(x@W_g) * (x@W_u) @ W_d``.

    ``activation`` ∈ {"silu", "gelu"} matching the LlamaConfig field.
    """
    return dispatch(
        "gated_mlp", x,
        gate_weight=gate_weight, up_weight=up_weight, down_weight=down_weight,
        activation=activation,
    )


__all__ = [
    "rmsnorm_qkv",
    "attention_output_proj",
    "gated_mlp",
]
