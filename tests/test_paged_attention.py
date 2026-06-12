"""Paged attention tests (M4)."""

import numpy as np
import pytest

import longhornai as lh
from longhornai.kernels.paged_attention.reference import ref_paged_attention
from longhornai.validation import assert_close, differential_test


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_paged_attention_matches_reference(rng, causal, dtype):
    B, S_q, H_q, H_kv, D = 2, 4, 4, 2, 8
    block_size = 4
    q = rng.standard_normal((B, S_q, H_q * D)).astype(dtype)
    ck, cv = lh.paged_kv_alloc(num_blocks=8, block_size=block_size,
                               num_kv_heads=H_kv, head_dim=D, dtype=dtype)
    block_table = np.array([[0, 1, -1], [2, 3, -1]], dtype=np.int32)
    k_new = rng.standard_normal((B, S_q, H_kv, D)).astype(dtype)
    v_new = rng.standard_normal((B, S_q, H_kv, D)).astype(dtype)
    seq_lens_before = np.zeros(B, dtype=np.int32)
    lh.paged_kv_append(ck, cv, k_new, v_new,
                       block_table=block_table, seq_lens=seq_lens_before,
                       block_size=block_size)
    seq_lens_after = np.full(B, S_q, dtype=np.int32)

    res = differential_test(
        "paged_attention", lh.paged_attention,
        q, ck, cv,
        dtype=dtype,
        block_table=block_table, seq_lens=seq_lens_after,
        block_size=block_size,
        num_q_heads=H_q, num_kv_heads=H_kv, head_dim=D, causal=causal,
    )
    assert res.passed, res


def test_paged_attention_decode_step(rng):
    """Single-token decode against a populated paged cache."""
    B, S_q, H_q, H_kv, D = 1, 1, 4, 2, 8
    block_size = 4
    seq_len = 6  # 2 blocks with this block_size
    ck, cv = lh.paged_kv_alloc(num_blocks=4, block_size=block_size,
                               num_kv_heads=H_kv, head_dim=D, dtype=np.float32)
    # Populate KV via a uniform-batch append.
    block_table = np.array([[0, 1, -1, -1]], dtype=np.int32)
    k_full = rng.standard_normal((1, seq_len, H_kv, D)).astype(np.float32)
    v_full = rng.standard_normal((1, seq_len, H_kv, D)).astype(np.float32)
    lh.paged_kv_append(ck, cv, k_full, v_full,
                       block_table=block_table,
                       seq_lens=np.zeros(1, dtype=np.int32),
                       block_size=block_size)

    # Decode-style query: 1 new token attending to all seq_len history.
    q = rng.standard_normal((B, S_q, H_q * D)).astype(np.float32)
    out = lh.paged_attention(q, ck, cv,
                             block_table=block_table,
                             seq_lens=np.array([seq_len + 1], dtype=np.int32),
                             block_size=block_size,
                             num_q_heads=H_q, num_kv_heads=H_kv, head_dim=D, causal=True)
    assert out.shape == (B, S_q, H_q * D)
    assert np.all(np.isfinite(out))


def test_paged_kv_append_writes_correct_slots(rng):
    block_size = 4
    ck, cv = lh.paged_kv_alloc(num_blocks=4, block_size=block_size,
                               num_kv_heads=1, head_dim=2, dtype=np.float32)
    block_table = np.array([[2, 3, -1]], dtype=np.int32)  # use blocks 2 and 3
    k_new = np.arange(8 * 1 * 2, dtype=np.float32).reshape(1, 8, 1, 2)
    v_new = -k_new
    lh.paged_kv_append(ck, cv, k_new, v_new,
                       block_table=block_table,
                       seq_lens=np.zeros(1, dtype=np.int32),
                       block_size=block_size)
    # First 4 tokens go into block 2, next 4 into block 3.
    assert_close(ck[2, 0], k_new[0, :4, 0], np.float32, name="block2_k")
    assert_close(ck[3, 0], k_new[0, 4:, 0], np.float32, name="block3_k")
    # Untouched blocks stay zero.
    assert np.all(ck[0] == 0) and np.all(ck[1] == 0)


def test_paged_attention_causal_first_query_attends_to_first_kv(rng):
    """With causal=True and S_q == seq_len, query 0 sees only KV[0]."""
    B, S, H_q, H_kv, D = 1, 4, 2, 1, 4
    block_size = 4
    ck, cv = lh.paged_kv_alloc(num_blocks=2, block_size=block_size,
                               num_kv_heads=H_kv, head_dim=D, dtype=np.float32)
    block_table = np.array([[0, -1]], dtype=np.int32)
    k = rng.standard_normal((1, S, H_kv, D)).astype(np.float32)
    v = rng.standard_normal((1, S, H_kv, D)).astype(np.float32)
    lh.paged_kv_append(ck, cv, k, v, block_table=block_table,
                       seq_lens=np.zeros(1, dtype=np.int32),
                       block_size=block_size)
    q = rng.standard_normal((B, S, H_q * D)).astype(np.float32)
    out = lh.paged_attention(q, ck, cv,
                             block_table=block_table,
                             seq_lens=np.array([S], dtype=np.int32),
                             block_size=block_size,
                             num_q_heads=H_q, num_kv_heads=H_kv, head_dim=D, causal=True)
    # Reshape output: (B, S, H_q, D); query 0 across both heads = v[0,0]
    out_h = out.reshape(B, S, H_q, D)
    expected_q0 = np.broadcast_to(v[0, 0:1, 0:1, :], (1, 1, H_q, D)).copy()
    assert_close(out_h[:, 0:1, :, :], expected_q0, np.float32, name="paged-causal-first")
