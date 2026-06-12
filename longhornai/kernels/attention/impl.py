"""CPU (NumPy) backend implementations for the attention family.

* ``sdpa`` — direct math form; correctness anchor.
* ``flash_attention_v1`` — IO-aware tiled forward via online softmax.
* ``flash_attention_v2`` — same forward with the v2 algorithmic differences:
  Q is the outer parallel axis (already true in v1) **and** K/V blocks
  strictly above the causal diagonal are skipped. Numerically identical to
  v1 / SDPA, faster on real hardware.
* ``multi_head_attention`` — generic per-head attention; reshape, replicate
  K/V for GQA, dispatch to the named flash impl, concat heads.

All accumulate in float64 on the CPU reference (anchor policy, PLAN.md §2.2).
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import _acc, cpu


@cpu.register("sdpa")
def sdpa(q, k, v, *, scale=None, causal=False):
    out_dt = np.result_type(q, k, v)
    acc = _acc(out_dt)
    qa, ka, va = q.astype(acc), k.astype(acc), v.astype(acc)
    head_dim = qa.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    scores = (qa @ np.swapaxes(ka, -1, -2)) * scale
    if causal:
        S_q, S_kv = scores.shape[-2], scores.shape[-1]
        i = np.arange(S_q).reshape(-1, 1)
        j = np.arange(S_kv).reshape(1, -1)
        scores = np.where(j > i, -np.inf, scores)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return (p @ va).astype(out_dt)


def _flash_streaming(q, k, v, *, scale, causal, block_q, block_kv,
                     skip_above_diagonal: bool):
    """Streaming-softmax core shared by v1 and v2.

    ``skip_above_diagonal`` is the v2 optimization — terminate the K/V scan
    early once the K block lies strictly past the last Q row's causal limit.
    Numerically a no-op (those blocks contribute -inf scores anyway), but
    cuts work proportionally to seq-length on long causal sequences.

    Q and K share head_dim D; V's head_dim D_v may differ (MLA — DeepSeek).
    Output shape matches ``(..., S_q, D_v)``.
    """
    out_dt = np.result_type(q, k, v)
    acc = _acc(out_dt)

    *lead, S_q, D = q.shape
    S_kv = k.shape[-2]
    D_v = v.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    leading_size = int(np.prod(lead)) if lead else 1
    Qf = q.astype(acc).reshape(leading_size, S_q, D)
    Kf = k.astype(acc).reshape(leading_size, S_kv, D)
    Vf = v.astype(acc).reshape(leading_size, S_kv, D_v)
    out = np.empty((leading_size, S_q, D_v), dtype=acc)

    for b in range(leading_size):
        Q, K, V = Qf[b], Kf[b], Vf[b]
        for i0 in range(0, S_q, block_q):
            i1 = min(i0 + block_q, S_q)
            Qi = Q[i0:i1]
            br = i1 - i0

            Oi = np.zeros((br, D_v), dtype=acc)
            mi = np.full(br, -np.inf, dtype=acc)
            li = np.zeros(br, dtype=acc)

            for j0 in range(0, S_kv, block_kv):
                if causal and skip_above_diagonal and j0 > i1 - 1:
                    break  # v2: every Q row in this block is below all of K[j0:]
                j1 = min(j0 + block_kv, S_kv)
                Kj, Vj = K[j0:j1], V[j0:j1]

                Sij = (Qi @ Kj.T) * scale
                if causal:
                    rows = np.arange(i0, i1).reshape(-1, 1)
                    cols = np.arange(j0, j1).reshape(1, -1)
                    Sij = np.where(cols > rows, -np.inf, Sij)

                m_block = Sij.max(axis=-1)
                m_new = np.maximum(mi, m_block)

                finite = np.isfinite(m_new)
                safe_diff_old = np.where(finite, mi - m_new, 0.0)
                safe_diff_new = np.where(finite[:, None], Sij - m_new[:, None], 0.0)
                alpha = np.where(finite, np.exp(safe_diff_old), 0.0)
                pij = np.where(finite[:, None], np.exp(safe_diff_new), 0.0)

                li = alpha * li + pij.sum(axis=-1)
                Oi = alpha[:, None] * Oi + pij @ Vj
                mi = m_new

            li_safe = np.where(li > 0, li, 1.0)
            out[b, i0:i1] = Oi / li_safe[:, None]

    return out.astype(out_dt).reshape(*lead, S_q, D_v)


@cpu.register("flash_attention_v1")
def flash_attention_v1(q, k, v, *, scale=None, causal=False,
                       block_q=64, block_kv=64):
    return _flash_streaming(q, k, v, scale=scale, causal=causal,
                            block_q=block_q, block_kv=block_kv,
                            skip_above_diagonal=False)


@cpu.register("flash_attention_v2")
def flash_attention_v2(q, k, v, *, scale=None, causal=False,
                       block_q=64, block_kv=64):
    return _flash_streaming(q, k, v, scale=scale, causal=causal,
                            block_q=block_q, block_kv=block_kv,
                            skip_above_diagonal=True)


@cpu.register("multi_head_attention")
def multi_head_attention(q, k, v, *, num_q_heads, num_kv_heads, head_dim,
                         causal=False, scale=None, attn_impl="flash_v2"):
    out_dt = np.result_type(q, k, v)
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

    # Re-enter dispatch so the active backend stays in control of the inner
    # attention call (CPU under CPU, RTL under RTL — keeps cross-target
    # equivalence transparent).
    from ...runtime import dispatch as _dispatch
    if attn_impl == "sdpa":
        out_h = _dispatch("sdpa", q_h, k_h, v_h, scale=scale, causal=causal)
    elif attn_impl == "flash_v1":
        out_h = _dispatch("flash_attention_v1", q_h, k_h, v_h,
                          scale=scale, causal=causal)
    elif attn_impl in ("flash_v2", "flash"):
        out_h = _dispatch("flash_attention_v2", q_h, k_h, v_h,
                          scale=scale, causal=causal)
    else:
        raise ValueError(f"unknown attn_impl '{attn_impl}'")

    return out_h.transpose(0, 2, 1, 3).reshape(B, S_q, num_q_heads * D).astype(out_dt)
