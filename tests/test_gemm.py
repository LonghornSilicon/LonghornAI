import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import assert_close, differential_test

from conftest import DTYPES


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(64, 128, 32), (1, 1, 1), (128, 256, 64)])
def test_gemm_matches_golden(rng, dtype, shape):
    m, k, n = shape
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    res = differential_test("gemm", lh.gemm, a, b, dtype=dtype)
    assert res.passed, res


def test_gemm_alpha_beta(rng):
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 4)).astype(np.float32)
    c = rng.standard_normal((8, 4)).astype(np.float32)
    out = lh.gemm(a, b, c=c, alpha=2.0, beta=0.5)
    assert_close(out, 2.0 * (a @ b) + 0.5 * c, np.float32, name="gemm_alpha_beta")


def test_gemm_shape_validation(rng):
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((9, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        lh.gemm(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_gemm(rng, dtype):
    a = rng.standard_normal((4, 16, 32)).astype(dtype)
    b = rng.standard_normal((4, 32, 8)).astype(dtype)
    res = differential_test("batched_gemm", lh.batched_gemm, a, b, dtype=dtype)
    assert res.passed, res


def test_grouped_gemm(rng):
    a_list = [rng.standard_normal((m, 8)).astype(np.float32) for m in (3, 5, 7)]
    b_list = [rng.standard_normal((8, n)).astype(np.float32) for n in (4, 6, 2)]
    outs = lh.grouped_gemm(a_list, b_list)
    assert len(outs) == 3
    for out, a, b in zip(outs, a_list, b_list):
        assert_close(out, a @ b, np.float32, name="grouped")


def test_tensor_contraction(rng):
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    b = rng.standard_normal((6, 7)).astype(np.float32)
    out = lh.tensor_contraction(a, b, "ijk,kl->ijl")
    assert_close(out, np.einsum("ijk,kl->ijl", a, b), np.float32, name="contraction")
