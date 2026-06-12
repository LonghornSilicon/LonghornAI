"""CPU implementations for the sparse-attention family.

Each kernel uses the same masked-SDPA core as the reference; lhsil's
implementations will issue tile-skipping launches that compute only the
non-masked blocks for big-S savings.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu
from .reference import _sdpa_with_mask


@cpu.register("sliding_window_attention")
def sliding_window_attention(q, k, v, *, window, scale=None):
    out_dt = np.result_type(q, k, v)
    acc = _acc(out_dt)
    qa, ka, va = q.astype(acc), k.astype(acc), v.astype(acc)
    S_q = qa.shape[-2]
    S_kv = ka.shape[-2]
    i = np.arange(S_q).reshape(-1, 1)
    j = np.arange(S_kv).reshape(1, -1)
    mask = (j > i) | (j < i - window + 1)
    return _sdpa_with_mask(qa, ka, va, mask, scale).astype(out_dt)


@cpu.register("block_sparse_attention")
def block_sparse_attention(q, k, v, *, block_mask, block_size,
                            scale=None, causal=False):
    out_dt = np.result_type(q, k, v)
    acc = _acc(out_dt)
    qa, ka, va = q.astype(acc), k.astype(acc), v.astype(acc)
    S_q = qa.shape[-2]
    S_kv = ka.shape[-2]
    full_mask = ~np.repeat(np.repeat(block_mask, block_size, axis=0),
                            block_size, axis=1)
    if causal:
        i = np.arange(S_q).reshape(-1, 1)
        j = np.arange(S_kv).reshape(1, -1)
        full_mask = full_mask | (j > i)
    return _sdpa_with_mask(qa, ka, va, full_mask, scale).astype(out_dt)


@cpu.register("dilated_attention")
def dilated_attention(q, k, v, *, dilation, scale=None):
    out_dt = np.result_type(q, k, v)
    acc = _acc(out_dt)
    qa, ka, va = q.astype(acc), k.astype(acc), v.astype(acc)
    S_q = qa.shape[-2]
    S_kv = ka.shape[-2]
    i = np.arange(S_q).reshape(-1, 1)
    j = np.arange(S_kv).reshape(1, -1)
    keep = (j <= i) & (((i - j) % dilation) == 0)
    return _sdpa_with_mask(qa, ka, va, ~keep, scale).astype(out_dt)
