"""KV-cache kernel family (PLAN.md §3 Phase 2 / M3).

The KV cache holds the key/value projections of every past token so decode
attention can be O(S) instead of O(S²). M3 ships:

* ``kv_cache_alloc`` — allocate dense ``(B, H_kv, S_max, D)`` storage.
* ``kv_cache_append`` — write new K/V at a sequence offset (per-batch).
* ``kv_cache_gather`` — read the valid prefix per batch.
* ``kv_cache_quantize_append`` / ``kv_cache_dequantize_gather`` — INT8
  symmetric per-(batch,head) quantization for the FP16-vs-INT8 footprint
  tradeoff (PLAN.md §3 Phase 2 — "FP16/FP8/INT8 cache dtypes").

Paged KV (block-table indirection over a paged pool) lands in M4 alongside
continuous batching (PLAN.md §3 Phase 3).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...runtime import dispatch


def kv_cache_alloc(
    *,
    batch: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
    dtype=np.float16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Allocate ``(cache_k, cache_v)`` of shape ``(B, H_kv, S_max, D)``.

    Pure-Python helper; doesn't dispatch. Returns zero-initialized arrays.
    """
    shape = (batch, num_kv_heads, max_seq_len, head_dim)
    return np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)


def kv_cache_append(
    cache_k: np.ndarray,
    cache_v: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    *,
    position: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Write ``k_new``/``v_new`` into the cache at sequence offset ``position``.

    ``cache_k``, ``cache_v``: ``(B, H_kv, S_max, D)``.
    ``k_new``, ``v_new``:     ``(B, H_kv, S_new, D)``.

    Returns the (possibly new) cache arrays — backends may return the same
    object after in-place mutation, or a fresh allocation.
    """
    return dispatch("kv_cache_append", cache_k, cache_v, k_new, v_new,
                    position=position)


def kv_cache_gather(
    cache_k: np.ndarray, cache_v: np.ndarray, *, length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read the valid ``[:length]`` prefix of the cache."""
    return dispatch("kv_cache_gather", cache_k, cache_v, length=length)


# --- INT8 quantized variants --------------------------------------------------

def kv_cache_alloc_int8(
    *,
    batch: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate INT8 quantized KV cache + per-(batch,head,seq) scales.

    Returns ``(cache_k_int8, scale_k, cache_v_int8, scale_v)``:

    * ``cache_k_int8`` / ``cache_v_int8``: ``(B, H_kv, S_max, D)`` of int8
    * ``scale_k`` / ``scale_v``:           ``(B, H_kv, S_max)`` of float32

    Per-token scales (one per ``(B, H_kv, S)`` triple) keep dynamic range
    locally tight for streamed token writes.
    """
    cache_shape = (batch, num_kv_heads, max_seq_len, head_dim)
    scale_shape = (batch, num_kv_heads, max_seq_len)
    return (
        np.zeros(cache_shape, dtype=np.int8),
        np.zeros(scale_shape, dtype=np.float32),
        np.zeros(cache_shape, dtype=np.int8),
        np.zeros(scale_shape, dtype=np.float32),
    )


def kv_cache_quantize_append(
    cache_k_int8: np.ndarray,
    scale_k: np.ndarray,
    cache_v_int8: np.ndarray,
    scale_v: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    *,
    position: int,
):
    """Symmetric-INT8 quantize ``k_new``, ``v_new`` and write at ``position``."""
    return dispatch(
        "kv_cache_quantize_append",
        cache_k_int8, scale_k, cache_v_int8, scale_v,
        k_new, v_new, position=position,
    )


def kv_cache_dequantize_gather(
    cache_k_int8: np.ndarray, scale_k: np.ndarray,
    cache_v_int8: np.ndarray, scale_v: np.ndarray,
    *, length: int, dtype=np.float16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``[:length]`` of the INT8 cache and dequantize to ``dtype``."""
    return dispatch(
        "kv_cache_dequantize_gather",
        cache_k_int8, scale_k, cache_v_int8, scale_v,
        length=length, dtype=dtype,
    )


__all__ = [
    "kv_cache_alloc",
    "kv_cache_append",
    "kv_cache_gather",
    "kv_cache_alloc_int8",
    "kv_cache_quantize_append",
    "kv_cache_dequantize_gather",
]
