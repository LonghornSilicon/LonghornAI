"""Paged-attention family kernels (PLAN.md §3 Phase 2 / M4).

The KV cache is stored as a pool of fixed-size **blocks**; each request gets
a **block table** mapping its logical token positions to physical block IDs.
Attention reads K/V via the block table, so per-request KV storage is
non-contiguous — eliminating the fragmentation that plagues dense pre-
allocated caches and unlocking high-throughput continuous batching (PLAN.md
§3 Phase 3).

M4 ships:

* ``paged_kv_alloc`` — allocate the paged pool.
* ``paged_kv_append`` — write new K/V tokens into the current logical
  positions per request, allocating physical blocks via the block table.
* ``paged_attention`` — variable-length attention reading K/V through the
  block table.

The block-pool layout is ``(num_blocks, num_kv_heads, block_size, head_dim)``.
Block size is typically 16 (lhsil) or 64 (lhsil large-context mode). Block
size is part of the contract — changing it changes the block table semantics
but not the numerical result.

The M4 exit gate (PLAN.md §8) is **E2E Llama decode under continuous
batching on FPGA**, plus a published tokens/sec baseline. The continuous-
batching scheduler in :mod:`longhornai.runtime.scheduler` consumes these
kernels.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...runtime import dispatch


def paged_kv_alloc(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=np.float16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Allocate the paged KV pool.

    Returns ``(cache_k, cache_v)`` of shape
    ``(num_blocks, num_kv_heads, block_size, head_dim)``.
    """
    shape = (num_blocks, num_kv_heads, block_size, head_dim)
    return np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)


def paged_kv_append(
    cache_k: np.ndarray,
    cache_v: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    *,
    block_table: np.ndarray,
    seq_lens: np.ndarray,
    block_size: int,
):
    """Write ``k_new``/``v_new`` into the paged pool per request.

    ``cache_k`` / ``cache_v``: ``(num_blocks, num_kv_heads, block_size, head_dim)``.
    ``k_new`` / ``v_new``:     ``(B, S_new, num_kv_heads, head_dim)`` —
        ``S_new`` is the same for every request in this call.
    ``block_table``: ``(B, max_blocks_per_seq)`` — physical block IDs.
    ``seq_lens``:    ``(B,)`` — sequence length **before** this append; the
        new tokens occupy ``[seq_lens[b], seq_lens[b] + S_new)``.
    ``block_size``:  the block dimension of the pool.

    The caller is responsible for ensuring ``block_table`` has been allocated
    enough physical blocks to cover ``seq_lens[b] + S_new`` per request — the
    scheduler in :mod:`longhornai.runtime.scheduler` does this.
    """
    return dispatch(
        "paged_kv_append", cache_k, cache_v, k_new, v_new,
        block_table=block_table, seq_lens=seq_lens, block_size=block_size,
    )


def paged_attention(
    q: np.ndarray,
    cache_k: np.ndarray,
    cache_v: np.ndarray,
    *,
    block_table: np.ndarray,
    seq_lens: np.ndarray,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    causal: bool = True,
) -> np.ndarray:
    """Variable-length attention reading paged KV.

    ``q``:           ``(B, S_q, num_q_heads * head_dim)`` — same ``S_q`` per request.
    ``cache_k`` / ``cache_v``: paged pool, ``(num_blocks, num_kv_heads, block_size, head_dim)``.
    ``block_table``: ``(B, max_blocks_per_seq)`` — same as for append.
    ``seq_lens``:    ``(B,)`` — total length **including** the new query tokens.

    For each request the kernel gathers KV for positions ``[0, seq_lens[b])``
    via the block table, replicates K/V for GQA, and runs causal attention.
    Output: ``(B, S_q, num_q_heads * head_dim)``.

    GQA / MQA come for free: ``num_kv_heads`` ≤ ``num_q_heads`` with the
    standard divisibility constraint.
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
    return dispatch(
        "paged_attention", q, cache_k, cache_v,
        block_table=block_table, seq_lens=seq_lens, block_size=block_size,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        scale=scale, causal=causal,
    )


__all__ = [
    "paged_kv_alloc",
    "paged_kv_append",
    "paged_attention",
]
