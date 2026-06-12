"""Quantization primitive tests (M5).

Round-trip accuracy + INT4 pack/unpack bit-exactness.
"""

import numpy as np
import pytest

from longhornai.quantization import (
    QuantParams,
    dequantize_q,
    pack_int4,
    quantize_asymmetric,
    quantize_groupwise,
    unpack_int4,
)


# --- asymmetric quant ----------------------------------------------------

def test_asymmetric_int8_round_trip_per_tensor(rng):
    x = rng.standard_normal((8, 16)).astype(np.float32)
    q, params = quantize_asymmetric(x, bits=8, axis=None)
    assert q.dtype == np.int32
    xr = dequantize_q(q, params)
    rel = np.abs(xr - x).max() / np.abs(x).max()
    # Per-tensor INT8 round-trip is bounded by ~1/128 ≈ 0.8% relative.
    assert rel < 0.02, f"asym int8 rel err = {rel}"


def test_asymmetric_int8_round_trip_per_channel(rng):
    x = rng.standard_normal((8, 32)).astype(np.float32)
    q, params = quantize_asymmetric(x, bits=8, axis=1)
    xr = dequantize_q(q, params)
    rel = np.abs(xr - x).max() / np.abs(x).max()
    # Per-channel is at least as accurate as per-tensor.
    assert rel < 0.02, f"asym int8 per-channel rel err = {rel}"


def test_asymmetric_uses_full_dynamic_range(rng):
    x = rng.uniform(0.0, 5.0, size=(8, 16)).astype(np.float32)  # one-sided
    q, params = quantize_asymmetric(x, bits=8, axis=None)
    # Asymmetric should hit both ends of [0, 255] for one-sided data.
    assert q.min() <= 5
    assert q.max() >= 250


# --- groupwise quant -----------------------------------------------------

def test_groupwise_int4_shapes(rng):
    W = rng.standard_normal((128, 32)).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
    assert q.shape == W.shape
    # 128 / 32 = 4 groups along axis 0.
    assert params.scale.shape == (4, 32)
    # INT4 symmetric values in [-8, 7].
    assert q.min() >= -8 and q.max() <= 7


def test_groupwise_int4_round_trip(rng):
    W = rng.standard_normal((64, 16)).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=16, axis=0)
    Wr = dequantize_q(q, params)
    # Per-group scale = amax/7. Worst-case quant error per element ≤ scale/2.
    # Group amax ~ 2-3 → scale ~ 0.3-0.4 → max abs error ≤ 0.2.
    abs_err = np.abs(Wr - W).max()
    assert abs_err < 0.4


def test_groupwise_int4_asymmetric_round_trip(rng):
    W = rng.standard_normal((64, 16)).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=16, axis=0, asymmetric=True)
    Wr = dequantize_q(q, params)
    # Asymmetric int4 uses 16 levels asymmetrically. Worst-case scale = range/15.
    abs_err = np.abs(Wr - W).max()
    assert abs_err < 0.4


def test_groupwise_rejects_non_divisible_K(rng):
    W = rng.standard_normal((30, 8)).astype(np.float32)  # 30 not divisible by 32
    with pytest.raises(ValueError, match="divisible"):
        quantize_groupwise(W, bits=4, group_size=32, axis=0)


# --- INT4 packing --------------------------------------------------------

def test_int4_pack_unpack_round_trips(rng):
    q = rng.integers(-8, 8, size=(64, 32)).astype(np.int8)
    packed = pack_int4(q, axis=0)
    assert packed.dtype == np.int8
    assert packed.shape == (32, 32)
    unpacked = unpack_int4(packed, axis=0)
    assert unpacked.dtype == np.int8
    assert np.array_equal(q, unpacked)


def test_int4_pack_handles_extremes():
    """-8 and 7 are the symmetric INT4 extremes; both must round-trip."""
    q = np.array([[-8, 7, -8, 7], [-1, 0, 1, -2]], dtype=np.int8)
    packed = pack_int4(q, axis=0)
    unpacked = unpack_int4(packed, axis=0)
    assert np.array_equal(q, unpacked)


def test_pack_int4_rejects_odd_axis_length():
    q = np.zeros((5, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="even"):
        pack_int4(q, axis=0)
