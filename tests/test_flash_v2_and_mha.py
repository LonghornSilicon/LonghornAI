"""FlashAttention v2 + multi_head_attention tests (M3).

PLAN.md §8 M3 exit gate: FA v2 numerical parity (latency + numerics) with
the SDPA reference. These tests enforce the numerical half — latency is
tracked by `benchmarks/shape_suite.py` once an attention shape suite lands
in M4.
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import assert_close, differential_test


# --- FlashAttention v2 numerical parity (M3 exit gate) -----------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 32, 16), (2, 1, 64, 16), (1, 4, 16, 8)])
def test_flash_v2_matches_golden(rng, dtype, causal, shape):
    q = rng.standard_normal(shape).astype(dtype)
    k = rng.standard_normal(shape).astype(dtype)
    v = rng.standard_normal(shape).astype(dtype)
    res = differential_test(
        "flash_attention_v2", lh.flash_attention_v2, q, k, v,
        dtype=dtype, causal=causal,
    )
    assert res.passed, res


@pytest.mark.parametrize("causal", [False, True])
def test_flash_v2_matches_v1_and_sdpa_directly(rng, causal):
    """All three impls must agree to fp32 precision on the same inputs."""
    shape = (2, 4, 64, 16)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    o_sdpa = lh.sdpa(q, k, v, causal=causal)
    o_v1 = lh.flash_attention_v1(q, k, v, causal=causal)
    o_v2 = lh.flash_attention_v2(q, k, v, causal=causal)
    assert_close(o_v1, o_sdpa, np.float32, name=f"v1↔sdpa causal={causal}")
    assert_close(o_v2, o_sdpa, np.float32, name=f"v2↔sdpa causal={causal}")
    assert_close(o_v2, o_v1, np.float32, name=f"v2↔v1 causal={causal}")


@pytest.mark.parametrize("block_q,block_kv", [(8, 8), (16, 32), (32, 8), (4, 16)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_v2_block_size_invariance(rng, block_q, block_kv, causal):
    shape = (1, 2, 48, 16)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    out_a = lh.flash_attention_v2(q, k, v, causal=causal,
                                  block_q=block_q, block_kv=block_kv)
    out_b = lh.flash_attention_v2(q, k, v, causal=causal,
                                  block_q=64, block_kv=64)
    assert_close(out_a, out_b, np.float32,
                 name=f"v2 block-invariance bq={block_q} bkv={block_kv}")


def test_flash_v2_causal_skip_does_not_change_output(rng):
    """With causal=True, v2 skips above-diagonal blocks but output is unchanged."""
    shape = (1, 1, 64, 8)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    # block_kv=8 with S_kv=64 means 8 K-blocks; v2 skips the strictly-above ones.
    o_v1 = lh.flash_attention_v1(q, k, v, causal=True, block_q=8, block_kv=8)
    o_v2 = lh.flash_attention_v2(q, k, v, causal=True, block_q=8, block_kv=8)
    assert_close(o_v2, o_v1, np.float32, name="v2 causal-skip equivalence")


# --- multi_head_attention (M3) ----------------------------------------------

@pytest.mark.parametrize("attn_impl", ["sdpa", "flash_v1", "flash_v2"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_mha_matches_reference(rng, attn_impl, dtype):
    B, S, Hq, D = 2, 16, 4, 8
    q = rng.standard_normal((B, S, Hq * D)).astype(dtype)
    k = rng.standard_normal((B, S, Hq * D)).astype(dtype)
    v = rng.standard_normal((B, S, Hq * D)).astype(dtype)
    res = differential_test(
        "multi_head_attention", lh.multi_head_attention, q, k, v,
        dtype=dtype,
        num_q_heads=Hq, num_kv_heads=Hq, head_dim=D,
        causal=True, attn_impl=attn_impl,
    )
    assert res.passed, res


def test_mqa_single_kv_head(rng):
    B, S, Hq, D = 1, 8, 4, 8
    q = rng.standard_normal((B, S, Hq * D)).astype(np.float32)
    k = rng.standard_normal((B, S, 1 * D)).astype(np.float32)
    v = rng.standard_normal((B, S, 1 * D)).astype(np.float32)
    out = lh.mqa(q, k, v, num_q_heads=Hq, head_dim=D, causal=True)
    assert out.shape == (B, S, Hq * D)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("Hq,Hkv", [(4, 2), (6, 2), (8, 1), (6, 3)])
def test_gqa_groups(rng, Hq, Hkv):
    B, S, D = 1, 12, 8
    q = rng.standard_normal((B, S, Hq * D)).astype(np.float32)
    k = rng.standard_normal((B, S, Hkv * D)).astype(np.float32)
    v = rng.standard_normal((B, S, Hkv * D)).astype(np.float32)
    out = lh.gqa(q, k, v, num_q_heads=Hq, num_kv_heads=Hkv, head_dim=D, causal=True)
    assert out.shape == (B, S, Hq * D)
    assert np.all(np.isfinite(out))


def test_mha_rejects_non_divisible_heads():
    B, S, Hq, Hkv, D = 1, 4, 5, 2, 8  # 5 not divisible by 2
    q = np.zeros((B, S, Hq * D), dtype=np.float32)
    k = np.zeros((B, S, Hkv * D), dtype=np.float32)
    v = np.zeros((B, S, Hkv * D), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple"):
        lh.multi_head_attention(
            q, k, v, num_q_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        )
