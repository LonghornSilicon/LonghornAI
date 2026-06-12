"""Sparse-attention family kernels (PLAN.md §3 Phase 6 / §8 M7).

Three patterns matter for production LLM serving:

* ``sliding_window_attention`` — Mistral / Gemma-2 alternating local
  attention. Each query at position ``i`` attends only to keys in
  ``[i - window + 1, i]`` (causal + window). Slashes long-context
  attention from O(S²) to O(S × window) with negligible quality loss.
* ``block_sparse_attention`` — fixed sparsity pattern over (Q-block,
  K-block) tiles (Longformer / BigBird family). The mask is a boolean
  block matrix; non-masked blocks compute SDPA, masked blocks contribute
  ``-inf`` scores.
* ``dilated_attention`` — strided sparse attention (Sparse Transformer
  family). Each query attends to keys at stride ``d`` going back; useful
  for long-context structured documents.

All three reduce to SDPA with a structured mask; the references encode
that reduction so backends can be validated against the dense math.
"""

from __future__ import annotations

import numpy as np

from ...runtime import dispatch


def sliding_window_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    window: int, scale: float | None = None,
) -> np.ndarray:
    """Causal sliding-window attention.

    Each query at position ``i`` attends to keys in ``[max(0, i - window + 1),
    i]``. ``q`` / ``k`` / ``v`` are ``(..., S, D)`` with the standard leading
    dims (batch + heads). Returns ``(..., S_q, D)``.

    PLAN.md §3 Phase 6 / Mistral, Gemma-2.
    """
    return dispatch(
        "sliding_window_attention", q, k, v,
        window=window, scale=scale,
    )


def block_sparse_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    block_mask: np.ndarray, block_size: int,
    scale: float | None = None, causal: bool = False,
) -> np.ndarray:
    """Block-sparse attention with a per-(Q-block, K-block) mask.

    ``block_mask``: ``(num_q_blocks, num_kv_blocks)`` boolean — ``True``
    means *attend*, ``False`` means *masked out*. ``S_q`` and ``S_kv``
    must be divisible by ``block_size``.
    """
    return dispatch(
        "block_sparse_attention", q, k, v,
        block_mask=block_mask, block_size=block_size,
        scale=scale, causal=causal,
    )


def dilated_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    dilation: int, scale: float | None = None,
) -> np.ndarray:
    """Dilated causal attention.

    Each query at position ``i`` attends to keys at positions ``i, i - d,
    i - 2d, …`` (down to 0). Sparse-Transformer family.
    """
    return dispatch(
        "dilated_attention", q, k, v,
        dilation=dilation, scale=scale,
    )


__all__ = [
    "sliding_window_attention",
    "block_sparse_attention",
    "dilated_attention",
]
