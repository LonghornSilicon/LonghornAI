"""Mixture-of-Experts kernel family (PLAN.md §3 Phase 5 / §8 M6).

Operators:

* ``moe_router(x, gate_weight)`` — project tokens to per-expert logits.
* ``moe_top_k(logits, k)`` — pick top-k experts per token, return their
  ids and softmax-renormalized weights.
* ``moe_dispatch(x, expert_ids)`` — permute tokens so all tokens routed
  to the same expert are contiguous; returns the permuted activations,
  the token offsets per expert (for grouped GEMM), and the recovery
  index (for the un-permute step).
* ``moe_combine(expert_outputs, weights, recovery, n_tokens)`` — un-permute
  expert outputs and weighted-sum them back to per-token form.

The expert-side compute itself is **grouped GEMM** (already in
``kernels/gemm``): each expert sees a variable number of tokens. M6's
expert-parallel split lives in :mod:`longhornai.runtime.expert_parallel`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...runtime import dispatch


def moe_router(x: np.ndarray, gate_weight: np.ndarray) -> np.ndarray:
    """Project ``x`` of shape ``(tokens, hidden)`` to ``(tokens, num_experts)``."""
    return dispatch("moe_router", x, gate_weight)


def moe_top_k(
    logits: np.ndarray, *, k: int, normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick top-``k`` experts per token.

    ``logits``: ``(tokens, num_experts)``.
    Returns ``(expert_ids, weights)`` both of shape ``(tokens, k)``:
    * ``expert_ids`` — int32 indices.
    * ``weights``    — float32; softmax over the *selected* logits when
      ``normalize=True`` (the standard Mixtral / Switch convention) so the
      per-token weights sum to 1.
    """
    return dispatch("moe_top_k", logits, k=k, normalize=normalize)


def moe_dispatch(
    x: np.ndarray, expert_ids: np.ndarray, *, num_experts: int,
):
    """Permute tokens so all tokens routed to the same expert are contiguous.

    ``x``: ``(tokens, hidden)``.
    ``expert_ids``: ``(tokens, k)`` — for top-k routing each token shows up
    ``k`` times in the permuted activations (once per chosen expert).

    Returns ``(permuted_x, expert_offsets, recovery)``:
    * ``permuted_x`` — ``(tokens * k, hidden)`` — tokens in expert-major order.
    * ``expert_offsets`` — ``(num_experts + 1,)`` — start row of each expert
      in ``permuted_x``; the difference is the per-expert token count for
      grouped GEMM.
    * ``recovery`` — ``(tokens * k,)`` — flat indices that map each row of
      ``permuted_x`` back to its ``(token, k)`` slot in ``expert_ids``.
    """
    return dispatch(
        "moe_dispatch", x, expert_ids, num_experts=num_experts,
    )


def moe_combine(
    expert_outputs: np.ndarray,
    weights: np.ndarray,
    recovery: np.ndarray,
    *,
    n_tokens: int,
    hidden: int,
) -> np.ndarray:
    """Un-permute ``expert_outputs`` and weighted-sum back to per-token form.

    ``expert_outputs``: ``(tokens * k, hidden)`` — same row order as the
        ``permuted_x`` produced by :func:`moe_dispatch`.
    ``weights``:        ``(tokens, k)`` — the top-k softmax weights.
    ``recovery``:       ``(tokens * k,)`` — flat indices into ``(token, k)``.
    """
    return dispatch(
        "moe_combine", expert_outputs, weights, recovery,
        n_tokens=n_tokens, hidden=hidden,
    )


__all__ = [
    "moe_router",
    "moe_top_k",
    "moe_dispatch",
    "moe_combine",
]
