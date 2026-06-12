"""Float64 references for the sparse-attention family.

Each variant reduces to SDPA-with-mask; the references compute that math
in float64 so the differential harness can validate backend implementations.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _sdpa_with_mask(q64, k64, v64, mask, scale):
    """SDPA with an additive ``-inf`` mask: ``mask[i, j] == True`` → masked out."""
    head_dim = q64.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    scores = (q64 @ np.swapaxes(k64, -1, -2)) * scale
    scores = np.where(mask, -np.inf, scores)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ v64


def ref_sliding_window_attention(q, k, v, *, window, scale=None):
    q64, k64, v64 = _f64(q), _f64(k), _f64(v)
    S_q = q64.shape[-2]
    S_kv = k64.shape[-2]
    i = np.arange(S_q).reshape(-1, 1)
    j = np.arange(S_kv).reshape(1, -1)
    # Mask out: future tokens (causal) AND tokens older than (i - window + 1).
    mask = (j > i) | (j < i - window + 1)
    return _sdpa_with_mask(q64, k64, v64, mask, scale).astype(q.dtype)


def ref_block_sparse_attention(
    q, k, v, *, block_mask, block_size, scale=None, causal=False,
):
    q64, k64, v64 = _f64(q), _f64(k), _f64(v)
    S_q = q64.shape[-2]
    S_kv = k64.shape[-2]
    if S_q % block_size != 0 or S_kv % block_size != 0:
        raise ValueError(
            f"block_sparse_attention: S_q ({S_q}) and S_kv ({S_kv}) must be "
            f"divisible by block_size ({block_size})"
        )
    nq_b = S_q // block_size
    nkv_b = S_kv // block_size
    if block_mask.shape != (nq_b, nkv_b):
        raise ValueError(
            f"block_sparse_attention: expected block_mask shape ({nq_b}, "
            f"{nkv_b}); got {block_mask.shape}"
        )
    # Expand block_mask to per-element. mask[i, j] = True means *masked out*,
    # so invert: drop where block_mask is False.
    full_mask = ~np.repeat(np.repeat(block_mask, block_size, axis=0),
                            block_size, axis=1)
    if causal:
        i = np.arange(S_q).reshape(-1, 1)
        j = np.arange(S_kv).reshape(1, -1)
        full_mask = full_mask | (j > i)
    return _sdpa_with_mask(q64, k64, v64, full_mask, scale).astype(q.dtype)


def ref_dilated_attention(q, k, v, *, dilation, scale=None):
    q64, k64, v64 = _f64(q), _f64(k), _f64(v)
    S_q = q64.shape[-2]
    S_kv = k64.shape[-2]
    i = np.arange(S_q).reshape(-1, 1)
    j = np.arange(S_kv).reshape(1, -1)
    # Causal: j <= i. Dilated: only positions where (i - j) is a multiple
    # of dilation; equivalently, j in {i, i-d, i-2d, ...}.
    causal_keep = j <= i
    aligned = ((i - j) % dilation) == 0
    keep = causal_keep & aligned
    mask = ~keep
    return _sdpa_with_mask(q64, k64, v64, mask, scale).astype(q.dtype)


REFERENCES = {
    "sliding_window_attention": ref_sliding_window_attention,
    "block_sparse_attention": ref_block_sparse_attention,
    "dilated_attention": ref_dilated_attention,
}

__all__ = [
    "REFERENCES",
    "ref_sliding_window_attention",
    "ref_block_sparse_attention",
    "ref_dilated_attention",
]
