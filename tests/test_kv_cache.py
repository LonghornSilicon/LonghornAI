"""KV-cache kernel tests (M3).

PLAN.md §3 Phase 2 / M3 — KV-cache kernels: alloc / append / gather, plus
the INT8 quantized round-trip. Append/gather are structural (the cache
state must equal the reference state); the INT8 round-trip is bounded by
the symmetric-INT8 representation error.
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.kernels.kv_cache.reference import (
    ref_kv_cache_append,
    ref_kv_cache_gather,
    ref_kv_cache_quantize_append,
    ref_kv_cache_dequantize_gather,
)
from longhornai.validation import assert_close


# --- dense FP cache --------------------------------------------------------

def test_alloc_shape_and_dtype():
    ck, cv = lh.kv_cache_alloc(
        batch=2, num_kv_heads=4, max_seq_len=32, head_dim=8, dtype=np.float16,
    )
    assert ck.shape == cv.shape == (2, 4, 32, 8)
    assert ck.dtype == cv.dtype == np.float16
    assert np.all(ck == 0) and np.all(cv == 0)


def test_append_writes_at_position(rng):
    ck, cv = lh.kv_cache_alloc(batch=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    k_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    v_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    lh.kv_cache_append(ck, cv, k_new, v_new, position=2)
    # Slots [2:5] hold k_new/v_new; surrounding slots untouched.
    assert_close(ck[..., 2:5, :], k_new, np.float16, name="append_k")
    assert_close(cv[..., 2:5, :], v_new, np.float16, name="append_v")
    assert np.all(ck[..., :2, :] == 0)
    assert np.all(ck[..., 5:, :] == 0)


def test_append_matches_reference(rng):
    ck0, cv0 = lh.kv_cache_alloc(batch=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    k_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    v_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    ck1, cv1 = lh.kv_cache_append(ck0.copy(), cv0.copy(), k_new, v_new, position=1)
    rk, rv = ref_kv_cache_append(ck0, cv0, k_new, v_new, position=1)
    assert_close(ck1, rk, np.float16, name="append_k_ref")
    assert_close(cv1, rv, np.float16, name="append_v_ref")


def test_gather_returns_prefix(rng):
    ck, cv = lh.kv_cache_alloc(batch=1, num_kv_heads=2, max_seq_len=8, head_dim=4)
    ck[...] = rng.standard_normal(ck.shape).astype(np.float16)
    cv[...] = rng.standard_normal(cv.shape).astype(np.float16)
    gk, gv = lh.kv_cache_gather(ck, cv, length=5)
    assert gk.shape == gv.shape == (1, 2, 5, 4)
    rk, rv = ref_kv_cache_gather(ck, cv, length=5)
    assert_close(gk, rk, np.float16, name="gather_k")
    assert_close(gv, rv, np.float16, name="gather_v")


def test_streaming_decode_writes_grow_cache(rng):
    """Mimic per-token decode: append 1 token at a time, gather grows."""
    ck, cv = lh.kv_cache_alloc(batch=1, num_kv_heads=1, max_seq_len=4, head_dim=4)
    tokens = []
    for t in range(4):
        k_new = rng.standard_normal((1, 1, 1, 4)).astype(np.float16)
        v_new = rng.standard_normal((1, 1, 1, 4)).astype(np.float16)
        tokens.append((k_new, v_new))
        lh.kv_cache_append(ck, cv, k_new, v_new, position=t)
        gk, gv = lh.kv_cache_gather(ck, cv, length=t + 1)
        assert gk.shape == (1, 1, t + 1, 4)
        # Last slot holds the just-appended token.
        assert_close(gk[..., -1, :], k_new[..., 0, :], np.float16,
                     name=f"streaming_token_{t}")


# --- INT8 quantized cache --------------------------------------------------

def test_int8_alloc_shapes():
    qk, sk, qv, sv = lh.kv_cache_alloc_int8(
        batch=2, num_kv_heads=4, max_seq_len=16, head_dim=8,
    )
    assert qk.shape == qv.shape == (2, 4, 16, 8)
    assert qk.dtype == qv.dtype == np.int8
    assert sk.shape == sv.shape == (2, 4, 16)
    assert sk.dtype == sv.dtype == np.float32


def test_int8_round_trip_within_bound(rng):
    """INT8 symmetric quant: relative round-trip error ≤ 1/127."""
    qk, sk, qv, sv = lh.kv_cache_alloc_int8(
        batch=1, num_kv_heads=2, max_seq_len=8, head_dim=4,
    )
    k_real = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    v_real = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    lh.kv_cache_quantize_append(qk, sk, qv, sv, k_real, v_real, position=0)
    dk, dv = lh.kv_cache_dequantize_gather(qk, sk, qv, sv, length=3, dtype=np.float32)
    # Per-token symmetric INT8 has relative error ≤ 1/127 ≈ 0.79% within range.
    rel_k = np.abs(dk - k_real).max() / np.abs(k_real).max()
    rel_v = np.abs(dv - v_real).max() / np.abs(v_real).max()
    assert rel_k < 0.01, f"INT8 round-trip rel err k = {rel_k}"
    assert rel_v < 0.01, f"INT8 round-trip rel err v = {rel_v}"


def test_int8_quantize_append_matches_reference(rng):
    qk, sk, qv, sv = lh.kv_cache_alloc_int8(
        batch=1, num_kv_heads=2, max_seq_len=8, head_dim=4,
    )
    k_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    v_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    qk1, sk1, qv1, sv1 = lh.kv_cache_quantize_append(
        qk.copy(), sk.copy(), qv.copy(), sv.copy(), k_new, v_new, position=2,
    )
    rqk, rsk, rqv, rsv = ref_kv_cache_quantize_append(
        qk, sk, qv, sv, k_new, v_new, position=2,
    )
    assert np.array_equal(qk1, rqk)
    assert np.array_equal(qv1, rqv)
    assert_close(sk1, rsk, np.float32, name="int8_scale_k")
    assert_close(sv1, rsv, np.float32, name="int8_scale_v")


def test_int8_dequantize_gather_returns_correct_dtype():
    qk, sk, qv, sv = lh.kv_cache_alloc_int8(
        batch=1, num_kv_heads=1, max_seq_len=4, head_dim=4,
    )
    dk, dv = lh.kv_cache_dequantize_gather(qk, sk, qv, sv, length=4, dtype=np.float16)
    assert dk.dtype == dv.dtype == np.float16
    dk32, _ = lh.kv_cache_dequantize_gather(qk, sk, qv, sv, length=4, dtype=np.float32)
    assert dk32.dtype == np.float32
