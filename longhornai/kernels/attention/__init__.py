"""Attention family kernels (PLAN.md §3 Phase 2).

Phase-2 operators landed across M2 and M3:

* M2: ``sdpa`` (math-form correctness anchor), ``flash_attention_v1``
  (IO-aware tiled forward).
* M3: ``flash_attention_v2`` (causal-skip + reordered tiling — the
  prefill-throughput target per PLAN.md §3 Phase 2 / §9.1),
  ``multi_head_attention`` and the named MHA/MQA/GQA wrappers.

Paged Attention lands in M4 alongside the decode/serving stack (PLAN.md §3
Phase 3). The M3 exit gate (PLAN.md §8) is **FA v2 parity** (latency +
numerics) and **RTL ≡ CPU on attention**.
"""

from __future__ import annotations

import numpy as np

from ...runtime import dispatch


# --- Bare attention kernels --------------------------------------------------

def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
         scale: float | None = None, causal: bool = False) -> np.ndarray:
    """Scaled dot-product attention (math form).

    Inputs are ``(..., S, D)`` — leading dims = batch + head dims, last two are
    sequence and head-dimension. ``scale`` defaults to ``1/sqrt(D)``.
    """
    return dispatch("sdpa", q, k, v, scale=scale, causal=causal)


def flash_attention_v1(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
                       scale: float | None = None, causal: bool = False,
                       block_q: int = 64, block_kv: int = 64) -> np.ndarray:
    """FlashAttention v1: IO-aware tiled forward via online softmax."""
    return dispatch("flash_attention_v1", q, k, v,
                    scale=scale, causal=causal,
                    block_q=block_q, block_kv=block_kv)


def flash_attention_v2(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
                       scale: float | None = None, causal: bool = False,
                       block_q: int = 64, block_kv: int = 64) -> np.ndarray:
    """FlashAttention v2: improved work partitioning + causal block-skip.

    Mathematically identical to v1; the algorithmic difference is internal
    (loop order and the "skip every K/V block strictly above the causal
    diagonal" optimization). The M3 exit gate enforces numerical parity with
    the SDPA reference.
    """
    return dispatch("flash_attention_v2", q, k, v,
                    scale=scale, causal=causal,
                    block_q=block_q, block_kv=block_kv)


# --- Multi-head attention wrappers (M3) --------------------------------------

def multi_head_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    num_q_heads: int, num_kv_heads: int, head_dim: int,
    causal: bool = False, scale: float | None = None,
    attn_impl: str = "flash_v2",
) -> np.ndarray:
    """Generic per-head attention dispatch.

    ``q`` is ``(B, S, H_q*D)``, ``k`` and ``v`` are ``(B, S_kv, H_kv*D)``. The
    impl reshapes to per-head form, replicates K/V along the head axis when
    ``H_kv < H_q`` (GQA / MQA), runs the named attention impl, and concats
    the heads back to ``(B, S, H_q*D)``.

    ``attn_impl`` ∈ {"sdpa", "flash_v1", "flash_v2"}.

    The named aliases ``mha`` / ``mqa`` / ``gqa`` are restrictions of this
    kernel along the ``num_kv_heads`` axis (PLAN.md §3 Phase 2).
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
    return dispatch(
        "multi_head_attention", q, k, v,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        causal=causal, scale=scale, attn_impl=attn_impl,
    )


def mha(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
        num_heads: int, head_dim: int, **kwargs) -> np.ndarray:
    """Multi-head attention: ``num_kv_heads == num_heads``. Llama-2, Qwen-1, etc."""
    return multi_head_attention(
        q, k, v,
        num_q_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim,
        **kwargs,
    )


def mqa(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
        num_q_heads: int, head_dim: int, **kwargs) -> np.ndarray:
    """Multi-query attention: a single shared KV head. Falcon, PaLM."""
    return multi_head_attention(
        q, k, v,
        num_q_heads=num_q_heads, num_kv_heads=1, head_dim=head_dim,
        **kwargs,
    )


def gqa(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
        num_q_heads: int, num_kv_heads: int, head_dim: int, **kwargs) -> np.ndarray:
    """Grouped-query attention: arbitrary KV-head grouping. Llama-3, Qwen-2, Mistral."""
    return multi_head_attention(
        q, k, v,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        **kwargs,
    )


__all__ = [
    "sdpa",
    "flash_attention_v1",
    "flash_attention_v2",
    "multi_head_attention",
    "mha",
    "mqa",
    "gqa",
]
