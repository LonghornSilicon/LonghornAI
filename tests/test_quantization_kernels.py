"""Quantized GEMM kernel tests (M5)."""

import numpy as np
import pytest

import longhornai as lh
from longhornai.quantization import (
    pack_int4,
    quantize_groupwise,
    scales_per_channel,
)
from longhornai.validation import assert_close, differential_test


# --- W8A8 ---------------------------------------------------------------

def test_w8a8_matches_reference(rng):
    A = rng.standard_normal((4, 16)).astype(np.float32)
    B = rng.standard_normal((16, 8)).astype(np.float32)
    sa = scales_per_channel(A, bits=8, axis=1)
    sb = scales_per_channel(B, bits=8, axis=0)
    aq = np.clip(np.rint(A / sa), -128, 127).astype(np.int8)
    bq = np.clip(np.rint(B / sb), -128, 127).astype(np.int8)
    res = differential_test(
        "gemm_w8a8", lh.gemm_w8a8, aq, bq,
        dtype=np.float32,
        scale_a=sa, scale_b=sb, out_dtype=np.float32,
    )
    assert res.passed, res


def test_w8a8_round_trip_within_int8_tolerance(rng):
    """W8A8 result should be within ~3% relative of the FP32 reference."""
    K = 64
    A = rng.standard_normal((4, K)).astype(np.float32)
    B = rng.standard_normal((K, 8)).astype(np.float32)
    sa = scales_per_channel(A, bits=8, axis=1)
    sb = scales_per_channel(B, bits=8, axis=0)
    aq = np.clip(np.rint(A / sa), -128, 127).astype(np.int8)
    bq = np.clip(np.rint(B / sb), -128, 127).astype(np.int8)
    out = lh.gemm_w8a8(aq, bq, scale_a=sa, scale_b=sb, out_dtype=np.float32)
    ref = A @ B
    # INT8 quant error per element ≤ scale/2; sum-of-K errors gives ~sqrt(K)*scale.
    # For K=64 with random N(0,1) inputs, expected abs err ~ 0.5.
    assert np.max(np.abs(out - ref)) < 1.5


def test_w8a8_asymmetric_path(rng):
    """Asymmetric W8A8 with zero-points matches the symmetric math when zp=0."""
    A = rng.standard_normal((4, 16)).astype(np.float32)
    B = rng.standard_normal((16, 8)).astype(np.float32)
    sa = scales_per_channel(A, bits=8, axis=1)
    sb = scales_per_channel(B, bits=8, axis=0)
    aq = np.clip(np.rint(A / sa), -128, 127).astype(np.int8)
    bq = np.clip(np.rint(B / sb), -128, 127).astype(np.int8)
    zp_zeros = np.zeros((1, 1), dtype=np.int32)
    out_sym = lh.gemm_w8a8(aq, bq, scale_a=sa, scale_b=sb, out_dtype=np.float32)
    out_asym = lh.gemm_w8a8(
        aq, bq, scale_a=sa, scale_b=sb,
        zero_point_a=zp_zeros, zero_point_b=zp_zeros, out_dtype=np.float32,
    )
    assert_close(out_asym, out_sym, np.float32, name="w8a8_asym_zp0")


# --- W4A16 --------------------------------------------------------------

def test_w4a16_matches_reference(rng):
    K, N = 64, 16
    W = rng.standard_normal((K, N)).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
    packed = pack_int4(q.astype(np.int8), axis=0)
    A = rng.standard_normal((4, K)).astype(np.float32)
    res = differential_test(
        "gemm_w4a16", lh.gemm_w4a16, A, packed,
        dtype=np.float32,
        scale_b=params.scale, group_size=32, K=K, out_dtype=np.float32,
    )
    assert res.passed, res


def test_w4a16_recovers_dense_within_int4_bound(rng):
    """Calibrated W4A16 vs FP32 dense — bounded by INT4 quant error scaling."""
    K, N = 128, 16
    W = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    q, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
    packed = pack_int4(q.astype(np.int8), axis=0)
    A = rng.standard_normal((8, K)).astype(np.float32)
    out = lh.gemm_w4a16(A, packed, scale_b=params.scale,
                         group_size=32, K=K, out_dtype=np.float32)
    ref = A @ W
    # 4 bits gives ≤ 1/16 relative quant resolution within a group; err on
    # output scales as sqrt(K) * scale * |A|.
    assert np.max(np.abs(out - ref)) < 5.0


def test_w4a16_dtype_propagation():
    K, N = 32, 8
    W = np.random.randn(K, N).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=16, axis=0)
    packed = pack_int4(q.astype(np.int8), axis=0)
    a16 = np.random.randn(2, K).astype(np.float16)
    out16 = lh.gemm_w4a16(a16, packed, scale_b=params.scale,
                           group_size=16, K=K)
    assert out16.dtype == np.float16
    a32 = np.random.randn(2, K).astype(np.float32)
    out32 = lh.gemm_w4a16(a32, packed, scale_b=params.scale,
                           group_size=16, K=K, out_dtype=np.float32)
    assert out32.dtype == np.float32


# --- activation quantize -----------------------------------------------

def test_activation_quantize_round_trip(rng):
    x = rng.standard_normal((4, 16)).astype(np.float32) * 2
    q, scale, zp = lh.activation_quantize(x, bits=8, axis=-1)
    assert q.dtype == np.int8
    assert zp is None
    xr = lh.activation_dequantize(q, scale=scale, dtype=np.float32)
    rel = np.abs(xr - x).max() / np.abs(x).max()
    assert rel < 0.02


def test_activation_quantize_asymmetric(rng):
    x = rng.uniform(0.0, 3.0, size=(4, 16)).astype(np.float32)  # one-sided
    q, scale, zp = lh.activation_quantize(x, bits=8, axis=-1, asymmetric=True)
    assert zp is not None
    # Asymmetric INT8 covers [0, 255] — carrier is uint8.
    assert q.dtype == np.uint8
    xr = lh.activation_dequantize(q, scale=scale, zero_point=zp, dtype=np.float32)
    rel = np.abs(xr - x).max() / np.abs(x).max()
    assert rel < 0.02
