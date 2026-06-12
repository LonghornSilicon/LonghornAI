"""Float64 references for the paged-attention family.

Paged append/attention reduce to dense KV append + SDPA per request. The
references encode that reduction so the differential harness can validate
that the paged path agrees with the dense path slot-for-slot.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _gather_paged_kv(cache, block_table, seq_len, block_size):
    """Gather a request's logical KV [0:seq_len] from the paged pool.

    cache:       (num_blocks, num_kv_heads, block_size, head_dim)
    block_table: (max_blocks,) — block IDs for one request
    seq_len:     scalar valid length
    Returns:     (num_kv_heads, seq_len, head_dim)
    """
    if seq_len == 0:
        H_kv, _, D = cache.shape[1], cache.shape[2], cache.shape[3]
        return np.empty((H_kv, 0, D), dtype=cache.dtype)
    n_blocks = (seq_len + block_size - 1) // block_size
    blocks = cache[block_table[:n_blocks]]              # (n_blocks, H_kv, B, D)
    blocks = np.transpose(blocks, (1, 0, 2, 3))         # (H_kv, n_blocks, B, D)
    H_kv, _, _, D = blocks.shape
    flat = blocks.reshape(H_kv, n_blocks * block_size, D)
    return flat[:, :seq_len, :]


def ref_paged_kv_append(
    cache_k, cache_v, k_new, v_new, *, block_table, seq_lens, block_size,
):
    """Reference: scatter k_new/v_new into the paged pool per request."""
    out_k = cache_k.copy()
    out_v = cache_v.copy()
    B, S_new, H_kv, D = k_new.shape
    for b in range(B):
        start = int(seq_lens[b])
        for t in range(S_new):
            pos = start + t
            block_idx = pos // block_size
            within = pos % block_size
            phys = int(block_table[b, block_idx])
            out_k[phys, :, within, :] = k_new[b, t]
            out_v[phys, :, within, :] = v_new[b, t]
    return out_k, out_v


def ref_paged_attention(
    q, cache_k, cache_v, *,
    block_table, seq_lens, block_size,
    num_q_heads, num_kv_heads, head_dim,
    scale=None, causal=True,
):
    """Reference: per-request gather + dense SDPA in float64."""
    q64 = _f64(q)
    ck64 = _f64(cache_k)
    cv64 = _f64(cache_v)
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    repeats = num_q_heads // num_kv_heads

    B, S_q, _ = q64.shape
    out = np.zeros((B, S_q, num_q_heads * head_dim), dtype=np.float64)

    for b in range(B):
        seq_len = int(seq_lens[b])
        # Reshape this batch's queries: (S_q, num_q_heads, head_dim)
        Qb = q64[b].reshape(S_q, num_q_heads, head_dim)
        # Gather paged KV for this request
        Kb = _gather_paged_kv(ck64, block_table[b], seq_len, block_size)
        Vb = _gather_paged_kv(cv64, block_table[b], seq_len, block_size)
        if repeats != 1:
            Kb = np.repeat(Kb, repeats, axis=0)  # (num_q_heads, seq_len, D)
            Vb = np.repeat(Vb, repeats, axis=0)
        # Attention per head: scores (num_q_heads, S_q, seq_len)
        Qbh = np.transpose(Qb, (1, 0, 2))        # (num_q_heads, S_q, D)
        scores = (Qbh @ np.swapaxes(Kb, -1, -2)) * scale
        if causal:
            # Queries are at positions [seq_len - S_q, seq_len). KV at [0, seq_len).
            q_pos = np.arange(seq_len - S_q, seq_len).reshape(1, -1, 1)
            kv_pos = np.arange(seq_len).reshape(1, 1, -1)
            scores = np.where(kv_pos > q_pos, -np.inf, scores)
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=-1, keepdims=True)
        out_h = p @ Vb                            # (num_q_heads, S_q, D)
        out_h = np.transpose(out_h, (1, 0, 2))    # (S_q, num_q_heads, D)
        out[b] = out_h.reshape(S_q, num_q_heads * head_dim)

    return out


REFERENCES = {
    "paged_kv_append": ref_paged_kv_append,
    "paged_attention": ref_paged_attention,
}

__all__ = [
    "REFERENCES",
    "ref_paged_kv_append",
    "ref_paged_attention",
    "_gather_paged_kv",
]
