"""Float64 golden references for the attention family.

The SDPA math IS the contract for every variant — ``flash_attention_v1``,
``flash_attention_v2``, and ``multi_head_attention`` all compute the same
mathematical function under different work partitionings. PLAN.md §8 M2/M3
exit gates require numerical parity with this reference.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def ref_sdpa(q, k, v, *, scale=None, causal=False):
    """Math-form scaled dot-product attention, computed in float64."""
    q, k, v = _f64(q), _f64(k), _f64(v)
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    if causal:
        S_q, S_kv = scores.shape[-2], scores.shape[-1]
        i = np.arange(S_q).reshape(-1, 1)
        j = np.arange(S_kv).reshape(1, -1)
        scores = np.where(j > i, -np.inf, scores)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ v


def ref_flash_attention_v1(q, k, v, *, scale=None, causal=False,
                           block_q=64, block_kv=64):
    return ref_sdpa(q, k, v, scale=scale, causal=causal)


def ref_flash_attention_v2(q, k, v, *, scale=None, causal=False,
                           block_q=64, block_kv=64):
    return ref_sdpa(q, k, v, scale=scale, causal=causal)


def ref_multi_head_attention(
    q, k, v, *, num_q_heads, num_kv_heads, head_dim,
    causal=False, scale=None, attn_impl="flash_v2",
):
    """Reference: split heads, replicate K/V, run SDPA, concat heads."""
    q, k, v = _f64(q), _f64(k), _f64(v)
    B, S_q, _ = q.shape
    S_kv = k.shape[1]
    D = head_dim

    q_h = q.reshape(B, S_q, num_q_heads, D).transpose(0, 2, 1, 3)
    k_h = k.reshape(B, S_kv, num_kv_heads, D).transpose(0, 2, 1, 3)
    v_h = v.reshape(B, S_kv, num_kv_heads, D).transpose(0, 2, 1, 3)
    if num_kv_heads != num_q_heads:
        repeats = num_q_heads // num_kv_heads
        k_h = np.repeat(k_h, repeats, axis=1)
        v_h = np.repeat(v_h, repeats, axis=1)
    out_h = ref_sdpa(q_h, k_h, v_h, scale=scale, causal=causal)
    return out_h.transpose(0, 2, 1, 3).reshape(B, S_q, num_q_heads * D)


REFERENCES = {
    "sdpa": ref_sdpa,
    "flash_attention_v1": ref_flash_attention_v1,
    "flash_attention_v2": ref_flash_attention_v2,
    "multi_head_attention": ref_multi_head_attention,
}

__all__ = [
    "REFERENCES",
    "ref_sdpa",
    "ref_flash_attention_v1",
    "ref_flash_attention_v2",
    "ref_multi_head_attention",
]
