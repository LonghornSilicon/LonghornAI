"""CPU (NumPy) implementations for the paged-attention family.

The block-pool layout is ``(num_blocks, num_kv_heads, block_size, head_dim)``.
The CPU impl scatters/gathers via plain fancy indexing — clear and correct,
not the perf path. lhsil's paged-attention kernel will issue gather DMAs
through the block table directly.

Append mutates the cache pool in place (matches the lhsil contract); attention
is read-only.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu
from .reference import _gather_paged_kv


@cpu.register("paged_kv_append")
def paged_kv_append(
    cache_k, cache_v, k_new, v_new, *, block_table, seq_lens, block_size,
):
    B, S_new = k_new.shape[0], k_new.shape[1]
    for b in range(B):
        start = int(seq_lens[b])
        for t in range(S_new):
            pos = start + t
            block_idx = pos // block_size
            within = pos % block_size
            phys = int(block_table[b, block_idx])
            cache_k[phys, :, within, :] = k_new[b, t]
            cache_v[phys, :, within, :] = v_new[b, t]
    return cache_k, cache_v


@cpu.register("paged_attention")
def paged_attention(
    q, cache_k, cache_v, *,
    block_table, seq_lens, block_size,
    num_q_heads, num_kv_heads, head_dim,
    scale=None, causal=True,
):
    out_dt = np.result_type(q, cache_k, cache_v)
    acc = _acc(out_dt)
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    repeats = num_q_heads // num_kv_heads

    B, S_q, _ = q.shape
    out = np.empty((B, S_q, num_q_heads * head_dim), dtype=acc)

    qa = q.astype(acc)
    cka = cache_k.astype(acc)
    cva = cache_v.astype(acc)

    for b in range(B):
        seq_len = int(seq_lens[b])
        Qb = qa[b].reshape(S_q, num_q_heads, head_dim)
        Kb = _gather_paged_kv(cka, block_table[b], seq_len, block_size)
        Vb = _gather_paged_kv(cva, block_table[b], seq_len, block_size)
        if repeats != 1:
            Kb = np.repeat(Kb, repeats, axis=0)
            Vb = np.repeat(Vb, repeats, axis=0)
        Qbh = np.transpose(Qb, (1, 0, 2))         # (num_q_heads, S_q, D)
        scores = (Qbh @ np.swapaxes(Kb, -1, -2)) * scale
        if causal:
            q_pos = np.arange(seq_len - S_q, seq_len).reshape(1, -1, 1)
            kv_pos = np.arange(seq_len).reshape(1, 1, -1)
            scores = np.where(kv_pos > q_pos, -np.inf, scores)
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=-1, keepdims=True)
        out_h = p @ Vb                             # (num_q_heads, S_q, D)
        out_h = np.transpose(out_h, (1, 0, 2))     # (S_q, num_q_heads, D)
        out[b] = out_h.reshape(S_q, num_q_heads * head_dim)

    return out.astype(out_dt)
