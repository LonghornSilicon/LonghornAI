"""CPU (NumPy) backend implementations for the KV-cache family.

The CPU impl mutates the supplied cache arrays in place when possible — that
matches the lhsil contract where the cache lives in device memory and is
updated by the append kernel without reallocation. The reference returns
fresh copies so the differential harness compares values, not aliasing.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import cpu
from .reference import _int8_dequantize_per_token, _int8_quantize_per_token


@cpu.register("kv_cache_append")
def kv_cache_append(cache_k, cache_v, k_new, v_new, *, position):
    s_new = k_new.shape[-2]
    cache_k[..., position : position + s_new, :] = k_new
    cache_v[..., position : position + s_new, :] = v_new
    return cache_k, cache_v


@cpu.register("kv_cache_gather")
def kv_cache_gather(cache_k, cache_v, *, length):
    # Returns views; callers that need an independent buffer should `.copy()`.
    return cache_k[..., :length, :], cache_v[..., :length, :]


@cpu.register("kv_cache_quantize_append")
def kv_cache_quantize_append(
    cache_k_int8, scale_k, cache_v_int8, scale_v,
    k_new, v_new, *, position,
):
    s_new = k_new.shape[-2]
    qk, sk_new = _int8_quantize_per_token(k_new)
    qv, sv_new = _int8_quantize_per_token(v_new)
    cache_k_int8[..., position : position + s_new, :] = qk
    cache_v_int8[..., position : position + s_new, :] = qv
    scale_k[..., position : position + s_new] = sk_new
    scale_v[..., position : position + s_new] = sv_new
    return cache_k_int8, scale_k, cache_v_int8, scale_v


@cpu.register("kv_cache_dequantize_gather")
def kv_cache_dequantize_gather(
    cache_k_int8, scale_k, cache_v_int8, scale_v, *, length, dtype=np.float16,
):
    qk = cache_k_int8[..., :length, :]
    qv = cache_v_int8[..., :length, :]
    sk = scale_k[..., :length]
    sv = scale_v[..., :length]
    return (
        _int8_dequantize_per_token(qk, sk, dtype=dtype),
        _int8_dequantize_per_token(qv, sv, dtype=dtype),
    )
