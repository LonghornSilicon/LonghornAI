import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import assert_close, differential_test

from conftest import DTYPES


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "op,fn,shape",
    [
        ("layernorm", lh.layernorm, (32, 64)),
        ("rmsnorm", lh.rmsnorm, (32, 64)),
        ("softmax", lh.softmax, (16, 128)),
        ("gelu", lh.gelu, (32, 64)),
        ("silu", lh.silu, (32, 64)),
        ("reduce", lh.reduce, (32, 64)),
    ],
)
def test_elementwise_matches_golden(rng, dtype, op, fn, shape):
    x = rng.standard_normal(shape).astype(dtype)
    res = differential_test(op, fn, x, dtype=dtype)
    assert res.passed, res


def test_layernorm_affine(rng):
    x = rng.standard_normal((8, 32)).astype(np.float32)
    w = rng.standard_normal((32,)).astype(np.float32)
    b = rng.standard_normal((32,)).astype(np.float32)
    out = lh.layernorm(x, weight=w, bias=b)
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    expected = (x - mean) / np.sqrt(var + 1e-5) * w + b
    assert_close(out, expected, np.float32, name="layernorm_affine")


def test_softmax_sums_to_one(rng):
    x = rng.standard_normal((10, 50)).astype(np.float32)
    out = lh.softmax(x)
    assert_close(out.sum(-1), np.ones(10), np.float32, name="softmax_sum")


def test_gelu_tanh_variant(rng):
    x = rng.standard_normal((16, 16)).astype(np.float32)
    res = differential_test("gelu_tanh", lambda v: lh.gelu(v, approximate="tanh"),
                            x, dtype=np.float32)
    assert res.passed, res


def test_rope_preserves_norm(rng):
    # Rotary embedding is a rotation: it must preserve per-position vector norm.
    x = rng.standard_normal((2, 8, 16)).astype(np.float32)
    out = lh.rope(x)
    assert_close(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1),
                 np.float32, name="rope_norm")


def test_rope_odd_dim_rejected(rng):
    x = rng.standard_normal((1, 4, 7)).astype(np.float32)
    with pytest.raises(ValueError):
        lh.rope(x)


def test_embedding_lookup(rng):
    table = rng.standard_normal((100, 8)).astype(np.float32)
    ids = np.array([[1, 5, 9], [0, 50, 99]])
    out = lh.embedding_lookup(table, ids)
    assert out.shape == (2, 3, 8)
    assert_close(out, table[ids], np.float32, name="embedding")


@pytest.mark.parametrize("op", ["sum", "max", "mean"])
def test_reduce_ops(rng, op):
    x = rng.standard_normal((8, 16)).astype(np.float32)
    out = lh.reduce(x, op=op)
    expected = getattr(np, op)(x, axis=-1)
    assert_close(out, expected, np.float32, name=f"reduce_{op}")
