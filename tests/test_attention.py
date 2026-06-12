"""Attention-family tests (M2).

Phase-2 exit-gate (PLAN.md §8 M2): "FlashAttention v1 numerical parity with
the SDPA reference on the benchmark causal/non-causal suite". The tests in
this file enforce that contract plus the structural invariants attention is
expected to satisfy (block-size invariance, default scale, causal correctness,
softmax sums to one).
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import assert_close, differential_test


# --- SDPA correctness against the float64 golden -----------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 16),     # single head, short seq
        (2, 4, 16, 32),    # batch + multi-head
        (1, 2, 64, 16),    # longer seq
    ],
)
def test_sdpa_matches_reference(rng, dtype, causal, shape):
    q = rng.standard_normal(shape).astype(dtype)
    k = rng.standard_normal(shape).astype(dtype)
    v = rng.standard_normal(shape).astype(dtype)
    res = differential_test("sdpa", lh.sdpa, q, k, v, dtype=dtype, causal=causal)
    assert res.passed, res


# --- FlashAttention v1 numerical parity (M2 exit gate) -----------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 32, 16), (2, 1, 64, 16), (1, 4, 16, 8)])
def test_flash_v1_matches_golden(rng, dtype, causal, shape):
    """Flash-v1 ↔ SDPA reference within the per-dtype tolerance policy."""
    q = rng.standard_normal(shape).astype(dtype)
    k = rng.standard_normal(shape).astype(dtype)
    v = rng.standard_normal(shape).astype(dtype)
    res = differential_test(
        "flash_attention_v1", lh.flash_attention_v1, q, k, v,
        dtype=dtype, causal=causal,
    )
    assert res.passed, res


@pytest.mark.parametrize("causal", [False, True])
def test_flash_v1_matches_sdpa_kernel_directly(rng, causal):
    """The two kernel paths must agree, not just each agree with the reference."""
    shape = (2, 4, 32, 16)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    o_sdpa = lh.sdpa(q, k, v, causal=causal)
    o_flash = lh.flash_attention_v1(q, k, v, causal=causal)
    assert_close(o_flash, o_sdpa, np.float32, name=f"sdpa↔flash causal={causal}")


# --- Block-size invariance ---------------------------------------------------

@pytest.mark.parametrize("block_q,block_kv", [(8, 8), (16, 32), (32, 8), (4, 16)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_v1_independent_of_block_size(rng, block_q, block_kv, causal):
    """Tile sizes must be a perf knob only; output is contractually fixed."""
    shape = (1, 2, 48, 16)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    out_a = lh.flash_attention_v1(q, k, v, causal=causal,
                                  block_q=block_q, block_kv=block_kv)
    out_b = lh.flash_attention_v1(q, k, v, causal=causal,
                                  block_q=64, block_kv=64)
    assert_close(out_a, out_b, np.float32, name=f"flash block-invariance bq={block_q} bkv={block_kv}")


# --- Structural invariants ---------------------------------------------------

def test_default_scale_is_one_over_sqrt_dhead(rng):
    """The standard transformer convention; explicit scale must agree with default."""
    shape = (1, 2, 8, 16)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    out_default = lh.sdpa(q, k, v)
    out_explicit = lh.sdpa(q, k, v, scale=1.0 / np.sqrt(16))
    assert_close(out_default, out_explicit, np.float32, name="default-scale")


def test_causal_first_token_attends_only_to_itself(rng):
    """With causal=True, row 0 of attention output must equal v[..., 0, :]."""
    shape = (1, 1, 6, 8)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    out = lh.sdpa(q, k, v, causal=True)
    assert_close(out[..., 0, :], v[..., 0, :], np.float32, name="causal-first-token")
    out_f = lh.flash_attention_v1(q, k, v, causal=True, block_q=2, block_kv=2)
    assert_close(out_f[..., 0, :], v[..., 0, :], np.float32, name="flash-causal-first-token")


def test_attention_with_uniform_scores_returns_uniform_average(rng):
    """If Q is zero, scores are zero, softmax is uniform, output = mean(V)."""
    B, H, S, D = 1, 2, 8, 16
    q = np.zeros((B, H, S, D), dtype=np.float32)
    k = rng.standard_normal((B, H, S, D)).astype(np.float32)
    v = rng.standard_normal((B, H, S, D)).astype(np.float32)
    out = lh.sdpa(q, k, v)
    expected = np.broadcast_to(v.mean(axis=-2, keepdims=True), out.shape).copy()
    assert_close(out, expected, np.float32, name="uniform-attention")
