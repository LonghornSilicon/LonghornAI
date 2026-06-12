"""bfloat16 path tests.

NumPy carries no native bf16 dtype, so values flow as float32 with the
mantissa truncated by :func:`to_bf16`. The kernels are otherwise unchanged
— what we are testing is the contract: bf16 inputs, kernel result agrees
with the float64 golden under the bf16 tolerance.
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import (
    BF16_DTYPE_NAME,
    differential_test,
    to_bf16,
    tolerance_for,
)


def test_to_bf16_zeros_low_mantissa_bits():
    x = np.array([1.0, 1.0 + 2 ** -8, 3.14159265, -0.0001234, -7.5], dtype=np.float32)
    bf = to_bf16(x)
    assert bf.dtype == np.float32
    # Every bf16 value has zeros in the bottom 16 bits of its fp32 carrier.
    assert np.all((bf.view(np.uint32) & 0xFFFF) == 0)


def test_to_bf16_preserves_specials():
    x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
    bf = to_bf16(x)
    assert np.isnan(bf[0])
    assert np.isposinf(bf[1])
    assert np.isneginf(bf[2])
    assert bf[3] == 0.0 and bf[4] == 0.0


def test_to_bf16_round_trip_within_tolerance(rng):
    x = rng.standard_normal(1024).astype(np.float32)
    bf = to_bf16(x)
    tol = tolerance_for(BF16_DTYPE_NAME)
    # bf16 has 7 explicit mantissa bits; relative error must fit the policy.
    rel_err = np.abs(bf - x) / np.maximum(np.abs(x), 1e-30)
    assert rel_err.max() <= tol.rtol


@pytest.mark.parametrize(
    "op,fn,shape",
    [
        ("layernorm", lh.layernorm, (8, 64)),
        ("rmsnorm", lh.rmsnorm, (8, 64)),
        ("softmax", lh.softmax, (8, 64)),
        ("silu", lh.silu, (8, 64)),
        ("gelu", lh.gelu, (8, 64)),
        ("reduce", lh.reduce, (8, 64)),
    ],
)
def test_bf16_elementwise_under_tolerance(rng, op, fn, shape):
    x = to_bf16(rng.standard_normal(shape).astype(np.float32))
    res = differential_test(op, fn, x, dtype=BF16_DTYPE_NAME)
    assert res.passed, res


@pytest.mark.parametrize("shape", [(32, 64, 16), (1, 4096, 4096)])
def test_bf16_gemm_under_tolerance(rng, shape):
    m, k, n = shape
    a = to_bf16(rng.standard_normal((m, k)).astype(np.float32))
    b = to_bf16(rng.standard_normal((k, n)).astype(np.float32))
    res = differential_test("gemm", lh.gemm, a, b, dtype=BF16_DTYPE_NAME)
    assert res.passed, res


def test_bf16_rope_under_tolerance(rng):
    x = to_bf16(rng.standard_normal((2, 8, 16)).astype(np.float32))
    res = differential_test("rope", lh.rope, x, dtype=BF16_DTYPE_NAME)
    assert res.passed, res


def test_bf16_label_recognized_by_tolerance():
    tol = tolerance_for(BF16_DTYPE_NAME)
    assert tol.rtol > 0 and tol.atol > 0
    # bf16 must be looser than fp16 (it has fewer mantissa bits).
    assert tol.rtol >= tolerance_for("float16").rtol
