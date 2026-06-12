"""MoE kernel + Mixtral forward + tensor/expert parallelism tests (M6)."""

import numpy as np
import pytest

import longhornai as lh
from longhornai.kernels.moe.reference import (
    ref_moe_combine,
    ref_moe_dispatch,
    ref_moe_router,
    ref_moe_top_k,
)
from longhornai.models import (
    LlamaConfig,
    init_mixtral_weights,
    init_random_weights,
    llama_forward,
    mixtral_8x7b_config,
    mixtral_forward,
    mixtral_toy_config,
)
from longhornai.runtime import (
    llama_forward_tp,
    mixtral_forward_ep,
    shard_llama_for_tp,
    shard_mixtral_for_ep,
)
from longhornai.validation import assert_close


# --- MoE kernels --------------------------------------------------------

def test_moe_router_matches_dense_matmul(rng):
    x = rng.standard_normal((6, 16)).astype(np.float32)
    W = rng.standard_normal((16, 4)).astype(np.float32)
    out = lh.moe_router(x, W)
    assert_close(out, x @ W, np.float32, name="router")


def test_moe_top_k_weights_sum_to_one(rng):
    logits = rng.standard_normal((8, 6)).astype(np.float32)
    expert_ids, weights = lh.moe_top_k(logits, k=2, normalize=True)
    assert expert_ids.shape == (8, 2)
    assert weights.shape == (8, 2)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


def test_moe_top_k_picks_highest_logits(rng):
    logits = np.array([[0.1, 5.0, 0.2, 4.0]], dtype=np.float32)
    expert_ids, _ = lh.moe_top_k(logits, k=2)
    assert set(int(i) for i in expert_ids[0]) == {1, 3}


def test_moe_dispatch_offsets_match_token_counts(rng):
    x = rng.standard_normal((5, 8)).astype(np.float32)
    expert_ids = np.array([
        [0, 1], [0, 2], [1, 2], [3, 0], [2, 3],
    ], dtype=np.int32)
    permuted, offsets, recovery = lh.moe_dispatch(x, expert_ids, num_experts=4)
    assert permuted.shape == (10, 8)
    counts = np.diff(offsets)
    # Expected counts: expert 0 → 3 (token 0, 1, 3), 1 → 2, 2 → 3, 3 → 2.
    assert counts.tolist() == [3, 2, 3, 2]


def test_moe_combine_round_trip_with_identity_experts(rng):
    """combine(identity_experts, weights) reduces to weighted sum of input rows."""
    T, H, K = 6, 8, 2
    x = rng.standard_normal((T, H)).astype(np.float32)
    expert_ids = np.array([[0, 1]] * T, dtype=np.int32)
    weights = np.full((T, K), 0.5, dtype=np.float32)
    permuted, offsets, recovery = lh.moe_dispatch(x, expert_ids, num_experts=2)
    # Identity expert → output = input.
    out = lh.moe_combine(permuted, weights, recovery, n_tokens=T, hidden=H)
    expected = x  # weights sum to 1, identity expert
    assert_close(out, expected, np.float32, name="combine_identity")


def test_moe_dispatch_recovery_reverses_permutation(rng):
    """Reconstructing expert_ids from recovery should match the original."""
    T, H, K = 5, 4, 2
    x = rng.standard_normal((T, H)).astype(np.float32)
    expert_ids = np.array([
        [0, 1], [3, 0], [2, 1], [1, 3], [0, 2],
    ], dtype=np.int32)
    permuted, offsets, recovery = lh.moe_dispatch(x, expert_ids, num_experts=4)
    # Reconstruct: for each permuted row, look up the (token, k) slot it came from.
    flat_ids = expert_ids.reshape(-1)
    for perm_row, original_slot in enumerate(recovery):
        token = int(original_slot) // K
        slot = int(original_slot) % K
        # Verify the permuted row matches x[token] (identity recovery).
        assert_close(permuted[perm_row], x[token], np.float32,
                     name=f"recovery_row{perm_row}")


# --- Mixtral end-to-end --------------------------------------------------

def test_mixtral_forward_shape_and_finite():
    config = mixtral_toy_config()
    weights = init_mixtral_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    out = mixtral_forward(ids, weights, config)
    assert out.shape == (1, 8, config.vocab_size)
    assert np.all(np.isfinite(out))


def test_mixtral_fp16_forward_finite():
    """M6 exit gate component: Mixtral decode correct in FP16."""
    config = mixtral_toy_config()
    weights = init_mixtral_weights(config, dtype=np.float16, seed=42, scale=0.02)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out = mixtral_forward(ids, weights, config)
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out)), "Mixtral fp16 forward must not overflow / NaN"


def test_mixtral_sdpa_flash_v2_parity():
    """Mixtral output is invariant to attention-impl choice."""
    config = mixtral_toy_config()
    weights = init_mixtral_weights(config, dtype=np.float32, seed=1)
    ids = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    out_sdpa = mixtral_forward(ids, weights, config, attn_impl=lh.sdpa)
    out_flash = mixtral_forward(ids, weights, config, attn_impl=lh.flash_attention_v2)
    assert_close(out_flash, out_sdpa, np.float32, name="mixtral_sdpa_flash")


def test_mixtral_reject_top_k_above_num_experts():
    from longhornai.models import MixtralConfig
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        MixtralConfig(
            vocab_size=8, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=4,
            num_local_experts=2, num_experts_per_tok=4,
        )


def test_mixtral_8x7b_real_dimensions():
    c = mixtral_8x7b_config()
    assert c.num_local_experts == 8
    assert c.num_experts_per_tok == 2
    assert c.hidden_size == 4096
    assert c.intermediate_size == 14336
    assert c.rope_theta == 1_000_000.0


# --- Tensor parallelism --------------------------------------------------

def _tp_test_config():
    return LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )


@pytest.mark.parametrize("num_ranks", [1, 2])
def test_tp_llama_matches_single_device(num_ranks):
    config = _tp_test_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_dense = llama_forward(ids, weights, config)
    tp_weights = shard_llama_for_tp(weights, num_ranks=num_ranks)
    out_tp = llama_forward_tp(ids, tp_weights, config)
    assert_close(out_tp, out_dense, np.float32,
                 name=f"tp_{num_ranks}_vs_dense")


def test_tp_rejects_indivisible_heads():
    """Forward must fail when num_heads can't be split evenly across ranks."""
    config = LlamaConfig(
        vocab_size=8, hidden_size=12, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=3,  # not divisible by 2
        num_key_value_heads=3, head_dim=4,
    )
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    tp_weights = shard_llama_for_tp(weights, num_ranks=2)
    ids = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="divisible"):
        llama_forward_tp(ids, tp_weights, config)


def test_tp_llama_fp16_finite():
    config = _tp_test_config()
    weights = init_random_weights(config, dtype=np.float16, seed=0, scale=0.02)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    tp_weights = shard_llama_for_tp(weights, num_ranks=2)
    out = llama_forward_tp(ids, tp_weights, config)
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out))


# --- Expert parallelism --------------------------------------------------

@pytest.mark.parametrize("num_ranks", [1, 2, 4])
def test_ep_mixtral_matches_single_device(num_ranks):
    """M6 exit gate: expert-parallel Mixtral matches single-device output."""
    config = mixtral_toy_config()  # 4 experts → 1/2/4 ranks all valid
    weights = init_mixtral_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_single = mixtral_forward(ids, weights, config)
    ep_weights = shard_mixtral_for_ep(weights, num_ranks=num_ranks)
    out_ep = mixtral_forward_ep(ids, ep_weights, config)
    assert_close(out_ep, out_single, np.float32,
                 name=f"ep_{num_ranks}_vs_single")


def test_ep_rejects_non_divisible_experts():
    config = mixtral_toy_config()  # 4 experts
    weights = init_mixtral_weights(config, dtype=np.float32, seed=0)
    with pytest.raises(ValueError, match="divisible"):
        shard_mixtral_for_ep(weights, num_ranks=3)


def test_ep_mixtral_fp16():
    config = mixtral_toy_config()
    weights = init_mixtral_weights(config, dtype=np.float16, seed=42, scale=0.02)
    ep_weights = shard_mixtral_for_ep(weights, num_ranks=2)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out = mixtral_forward_ep(ids, ep_weights, config)
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out))


def test_ep_per_rank_expert_count():
    """Each rank holds exactly num_local_experts / num_ranks experts."""
    config = mixtral_toy_config()
    weights = init_mixtral_weights(config, dtype=np.float32, seed=0)
    ep = shard_mixtral_for_ep(weights, num_ranks=2)
    assert ep.experts_per_rank == config.num_local_experts // 2
    for layer in ep.layers:
        assert len(layer.experts) == 2  # num_ranks
        for rank_experts in layer.experts:
            assert len(rank_experts) == ep.experts_per_rank
