"""Float64 / structural references for the KV-cache family.

KV-cache append/gather are not arithmetic kernels — their contract is
structural (which slots get written, which get read). The references below
encode that structure so the differential harness can validate per-position
correctness and the INT8 round-trip accuracy.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def ref_kv_cache_append(cache_k, cache_v, k_new, v_new, *, position):
    """Reference: write k_new/v_new at [position : position + S_new]."""
    out_k = cache_k.copy()
    out_v = cache_v.copy()
    s_new = k_new.shape[-2]
    out_k[..., position : position + s_new, :] = k_new
    out_v[..., position : position + s_new, :] = v_new
    return out_k, out_v


def ref_kv_cache_gather(cache_k, cache_v, *, length):
    """Reference: slice the valid prefix [:length]."""
    return cache_k[..., :length, :].copy(), cache_v[..., :length, :].copy()


def _int8_quantize_per_token(x: np.ndarray):
    """Symmetric INT8 quantize per-(B, H, S) — one scale per token row."""
    # x: (B, H, S, D)
    amax = np.maximum(np.max(np.abs(x), axis=-1), 1e-8)  # (B, H, S)
    scale = (amax / 127.0).astype(np.float32)  # per-token scale
    q = np.rint(_f64(x) / scale[..., None]).clip(-128, 127).astype(np.int8)
    return q, scale


def _int8_dequantize_per_token(q: np.ndarray, scale: np.ndarray, *, dtype):
    """Inverse of :func:`_int8_quantize_per_token`."""
    return (q.astype(np.float32) * scale[..., None]).astype(dtype)


def ref_kv_cache_quantize_append(
    cache_k_int8, scale_k, cache_v_int8, scale_v,
    k_new, v_new, *, position,
):
    """Reference: per-token quantize + write."""
    out_k = cache_k_int8.copy()
    out_v = cache_v_int8.copy()
    sk = scale_k.copy()
    sv = scale_v.copy()
    s_new = k_new.shape[-2]
    qk, sk_new = _int8_quantize_per_token(k_new)
    qv, sv_new = _int8_quantize_per_token(v_new)
    out_k[..., position : position + s_new, :] = qk
    out_v[..., position : position + s_new, :] = qv
    sk[..., position : position + s_new] = sk_new
    sv[..., position : position + s_new] = sv_new
    return out_k, sk, out_v, sv


def ref_kv_cache_dequantize_gather(
    cache_k_int8, scale_k, cache_v_int8, scale_v, *, length, dtype,
):
    """Reference: read [:length] and dequantize per-token."""
    qk = cache_k_int8[..., :length, :]
    qv = cache_v_int8[..., :length, :]
    sk = scale_k[..., :length]
    sv = scale_v[..., :length]
    return (
        _int8_dequantize_per_token(qk, sk, dtype=dtype),
        _int8_dequantize_per_token(qv, sv, dtype=dtype),
    )


REFERENCES = {
    "kv_cache_append": ref_kv_cache_append,
    "kv_cache_gather": ref_kv_cache_gather,
    "kv_cache_quantize_append": ref_kv_cache_quantize_append,
    "kv_cache_dequantize_gather": ref_kv_cache_dequantize_gather,
}

__all__ = [
    "REFERENCES",
    "ref_kv_cache_append",
    "ref_kv_cache_gather",
    "ref_kv_cache_quantize_append",
    "ref_kv_cache_dequantize_gather",
]
