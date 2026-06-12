"""Float64 references for the fused-kernel family.

Each reference *un-fuses* the kernel into the canonical chain of standard
ops; the fused backend impl must produce the same output bit-tight under
the per-dtype tolerance policy.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


# --- rmsnorm + qkv ---------------------------------------------------------

def ref_rmsnorm_qkv(x, *, norm_weight, q_weight, k_weight, v_weight, eps=1e-5):
    xa = _f64(x)
    ms = np.mean(np.square(xa), axis=-1, keepdims=True)
    h = (xa / np.sqrt(ms + eps)) * _f64(norm_weight)
    out_dt = x.dtype
    q = (h @ _f64(q_weight)).astype(out_dt)
    k = (h @ _f64(k_weight)).astype(out_dt)
    v = (h @ _f64(v_weight)).astype(out_dt)
    return q, k, v


# --- attention + o_proj ----------------------------------------------------

def ref_attention_output_proj(q, k, v, *, o_weight, causal=False, scale=None):
    qa, ka, va = _f64(q), _f64(k), _f64(v)
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
    attn = p @ va                                              # (B, n_heads, S, D)
    # Concatenate heads → (B, S, n_heads * D), then project.
    B, n_heads, S, D = attn.shape
    flat = attn.transpose(0, 2, 1, 3).reshape(B * S, n_heads * D)
    out = flat @ _f64(o_weight)
    return out.reshape(B, S, -1).astype(q.dtype)


# --- gated MLP -------------------------------------------------------------

def _act(x, kind):
    if kind == "silu":
        return x / (1.0 + np.exp(-x))
    if kind == "gelu":
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    raise ValueError(f"unknown activation '{kind}'")


def ref_gated_mlp(x, *, gate_weight, up_weight, down_weight, activation="silu"):
    xa = _f64(x)
    g = _act(xa @ _f64(gate_weight), activation)
    u = xa @ _f64(up_weight)
    return ((g * u) @ _f64(down_weight)).astype(x.dtype)


REFERENCES = {
    "rmsnorm_qkv": ref_rmsnorm_qkv,
    "attention_output_proj": ref_attention_output_proj,
    "gated_mlp": ref_gated_mlp,
}

__all__ = [
    "REFERENCES",
    "ref_rmsnorm_qkv",
    "ref_attention_output_proj",
    "ref_gated_mlp",
]
