"""M7 exit-gate tests — silicon smoke + cross-target equivalence.

PLAN.md §8 M7 exit gate: **silicon smoke + per-kernel correctness pass;
cross-target equivalence holds**. The harness in
``validation/silicon_smoke.py`` runs every smoke fixture under each
registered backend; these tests pin the contract.

Plus the M7 deliverables that aren't kernel-level:
* DeepSeek-V2 forward (MLA + fine-grained MoE)
* Sparse-attention degeneration to dense SDPA at full density
* Fused kernels match un-fused chains
* Structured 2:4 sparsity
* Pipeline parallelism matches single-device output
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.models import (
    DeepSeekConfig,
    LlamaConfig,
    deepseek_forward,
    deepseek_toy_config,
    deepseek_v2_config,
    init_deepseek_weights,
    init_random_weights,
    llama_forward,
)
from longhornai.quantization import gemm_sparse_2to4, prune_2to4
from longhornai.runtime import (
    llama_forward_pp,
    micro_batch_forward_pp,
    shard_llama_for_pp,
)
from longhornai.runtime import get_backend
from longhornai.validation import (
    assert_close,
    run_full_cross_target_sweep,
    run_silicon_smoke,
)


# --- silicon smoke + cross-target equivalence (M7 exit gate) -------------

def test_lhsil_backend_registered():
    backends = lh.available_backends()
    assert "lhsil" in backends
    lhsil = get_backend("lhsil")
    cpu = get_backend("cpu")
    # The lhsil shim mirrors the full CPU op surface.
    assert set(lhsil.ops()) == set(cpu.ops())


def test_silicon_smoke_passes_on_lhsil():
    """M7 exit gate: every smoke fixture passes under lhsil."""
    report = run_silicon_smoke(backend="lhsil", reference="cpu")
    assert report.passed, report.report()
    # Smoke covers a meaningful slice of the kernel surface.
    assert len(report.outcomes) >= 15


def test_silicon_smoke_passes_on_every_pre_silicon_backend():
    """The same smoke set passes on cpu / sim / rtl / fpga / lhsil."""
    for backend in ("cpu", "sim", "rtl", "fpga", "lhsil"):
        report = run_silicon_smoke(backend=backend, reference="cpu")
        assert report.passed, f"{backend}: {report.report()}"


def test_full_cross_target_equivalence_holds():
    """PLAN.md §5.2 keystone — cross-target equivalence on every op."""
    sweep = run_full_cross_target_sweep()
    failures = [op for op, eq in sweep.items() if not eq.passed]
    assert not failures, f"failed ops: {failures}"


# --- DeepSeek-V2 forward -------------------------------------------------

def test_deepseek_forward_shape_and_finite():
    config = deepseek_toy_config()
    weights = init_deepseek_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    out = deepseek_forward(ids, weights, config)
    assert out.shape == (1, 8, config.vocab_size)
    assert np.all(np.isfinite(out))


def test_deepseek_fp16_forward_finite():
    config = deepseek_toy_config()
    weights = init_deepseek_weights(config, dtype=np.float16, seed=42, scale=0.02)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out = deepseek_forward(ids, weights, config)
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out)), "DeepSeek fp16 forward must not overflow / NaN"


def test_deepseek_sdpa_flash_v2_parity():
    """Output is invariant to attention-impl choice, even with MLA's
    asymmetric QK / V head dims."""
    config = deepseek_toy_config()
    weights = init_deepseek_weights(config, dtype=np.float32, seed=1)
    ids = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    out_sdpa = deepseek_forward(ids, weights, config, attn_impl=lh.sdpa)
    out_flash = deepseek_forward(ids, weights, config, attn_impl=lh.flash_attention_v2)
    assert_close(out_flash, out_sdpa, np.float32, name="deepseek_sdpa_flash")


def test_deepseek_uses_mla_dimensions():
    config = deepseek_toy_config()
    assert config.q_lora_rank > 0
    assert config.kv_lora_rank > 0
    assert config.qk_head_dim == config.qk_nope_head_dim + config.qk_rope_head_dim


def test_deepseek_v2_real_dimensions():
    c = deepseek_v2_config()
    assert c.q_lora_rank == 1536
    assert c.kv_lora_rank == 512
    assert c.qk_head_dim == 192
    assert c.num_routed_experts == 160
    assert c.num_shared_experts == 2
    assert c.num_experts_per_tok == 6


def test_deepseek_rejects_top_k_above_routed():
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        DeepSeekConfig(
            vocab_size=8, hidden_size=8, intermediate_size=4,
            num_hidden_layers=1, num_attention_heads=2,
            q_lora_rank=4, kv_lora_rank=4,
            qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2,
            num_routed_experts=2, num_shared_experts=1,
            num_experts_per_tok=4,
        )


def test_deepseek_with_flash_v1_works_too():
    """Flash-v1 also handles asymmetric QK / V head dims after the M7 fix."""
    config = deepseek_toy_config()
    weights = init_deepseek_weights(config, dtype=np.float32, seed=2)
    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    out = deepseek_forward(ids, weights, config, attn_impl=lh.flash_attention_v1)
    assert out.shape == (1, 4, config.vocab_size)
    assert np.all(np.isfinite(out))


# --- Sparse attention ----------------------------------------------------

def test_sliding_window_degenerates_to_full_causal():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
    out_sw = lh.sliding_window_attention(q, k, v, window=8)
    out_dense = lh.sdpa(q, k, v, causal=True)
    assert_close(out_sw, out_dense, np.float32, name="sw=full vs causal")


def test_sliding_window_masks_distant_keys():
    """Query 0 with window=1 sees only key 0 → output equals v[..., 0, :]."""
    rng = np.random.default_rng(1)
    q = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    k = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    v = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    out = lh.sliding_window_attention(q, k, v, window=1)
    # window=1 + causal: every query sees only its own position.
    for i in range(4):
        assert_close(out[..., i, :], v[..., i, :], np.float32,
                     name=f"sw=1 row {i}")


def test_block_sparse_dense_mask_equals_sdpa(rng):
    q = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
    v = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
    mask = np.ones((2, 2), dtype=bool)
    out = lh.block_sparse_attention(q, k, v, block_mask=mask, block_size=4)
    expected = lh.sdpa(q, k, v)
    assert_close(out, expected, np.float32, name="block-sparse all-true")


def test_dilated_dilation_one_equals_causal(rng):
    q = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    k = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    v = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    out = lh.dilated_attention(q, k, v, dilation=1)
    expected = lh.sdpa(q, k, v, causal=True)
    assert_close(out, expected, np.float32, name="dilated d=1")


# --- Fused kernels -------------------------------------------------------

def test_gated_mlp_silu_matches_unfused_chain(rng):
    H, I = 16, 32
    x = rng.standard_normal((4, H)).astype(np.float32)
    Wg = rng.standard_normal((H, I)).astype(np.float32)
    Wu = rng.standard_normal((H, I)).astype(np.float32)
    Wd = rng.standard_normal((I, H)).astype(np.float32)
    fused = lh.gated_mlp(x, gate_weight=Wg, up_weight=Wu, down_weight=Wd,
                         activation="silu")
    unfused = (lh.silu(x @ Wg) * (x @ Wu)) @ Wd
    assert_close(fused, unfused, np.float32, name="gated_mlp_silu")


def test_gated_mlp_gelu_matches_unfused_chain(rng):
    H, I = 16, 32
    x = rng.standard_normal((4, H)).astype(np.float32)
    Wg = rng.standard_normal((H, I)).astype(np.float32)
    Wu = rng.standard_normal((H, I)).astype(np.float32)
    Wd = rng.standard_normal((I, H)).astype(np.float32)
    fused = lh.gated_mlp(x, gate_weight=Wg, up_weight=Wu, down_weight=Wd,
                         activation="gelu")
    unfused = (lh.gelu(x @ Wg, approximate="tanh") * (x @ Wu)) @ Wd
    assert_close(fused, unfused, np.float32, name="gated_mlp_gelu")


def test_rmsnorm_qkv_matches_unfused_chain(rng):
    H = 16
    x = rng.standard_normal((4, H)).astype(np.float32)
    norm_w = np.ones(H, dtype=np.float32)
    Wq = rng.standard_normal((H, 16)).astype(np.float32)
    Wk = rng.standard_normal((H, 8)).astype(np.float32)
    Wv = rng.standard_normal((H, 8)).astype(np.float32)
    q_f, k_f, v_f = lh.rmsnorm_qkv(
        x, norm_weight=norm_w, q_weight=Wq, k_weight=Wk, v_weight=Wv,
    )
    h = lh.rmsnorm(x, weight=norm_w)
    assert_close(q_f, h @ Wq, np.float32, name="rmsnorm_qkv_q")
    assert_close(k_f, h @ Wk, np.float32, name="rmsnorm_qkv_k")
    assert_close(v_f, h @ Wv, np.float32, name="rmsnorm_qkv_v")


def test_attention_output_proj_matches_unfused_chain(rng):
    B, H_h, S, D = 1, 2, 8, 4
    H = H_h * D
    q = rng.standard_normal((B, H_h, S, D)).astype(np.float32)
    k = rng.standard_normal((B, H_h, S, D)).astype(np.float32)
    v = rng.standard_normal((B, H_h, S, D)).astype(np.float32)
    Wo = rng.standard_normal((H, 16)).astype(np.float32)
    fused = lh.attention_output_proj(q, k, v, o_weight=Wo, causal=True)
    attn = lh.sdpa(q, k, v, causal=True)
    flat = attn.transpose(0, 2, 1, 3).reshape(B * S, H)
    unfused = (flat @ Wo).reshape(B, S, 16)
    assert_close(fused, unfused, np.float32, name="attn_o_proj")


# --- Structured 2:4 sparsity --------------------------------------------

def test_prune_2to4_keeps_exactly_two_of_four(rng):
    W = rng.standard_normal((16, 8)).astype(np.float32)
    pruned, mask = prune_2to4(W, axis=0)
    # 50% kept overall; per-group exactly 2 of 4 kept along axis 0.
    assert mask.mean() == 0.5
    grouped = mask.reshape(4, 4, 8)
    assert np.all(grouped.sum(axis=1) == 2)


def test_prune_2to4_zeroes_dropped_entries(rng):
    W = rng.standard_normal((16, 8)).astype(np.float32)
    pruned, mask = prune_2to4(W, axis=0)
    assert np.all(pruned[~mask] == 0)


def test_gemm_sparse_2to4_matches_dense_matmul_on_pruned(rng):
    W = rng.standard_normal((32, 16)).astype(np.float32)
    pruned, _ = prune_2to4(W, axis=0)
    A = rng.standard_normal((4, 32)).astype(np.float32)
    out = gemm_sparse_2to4(A, pruned)
    ref = A @ pruned
    assert_close(out, ref, np.float32, name="2to4_vs_dense")


# --- Pipeline parallelism -----------------------------------------------

def _pp_test_config():
    return LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )


@pytest.mark.parametrize("num_ranks", [1, 2, 4])
def test_pp_llama_matches_single_device(num_ranks):
    config = _pp_test_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_dense = llama_forward(ids, weights, config)
    pp = shard_llama_for_pp(weights, num_ranks=num_ranks)
    out_pp = llama_forward_pp(ids, pp, config)
    assert_close(out_pp, out_dense, np.float32,
                 name=f"pp_{num_ranks}_vs_dense")


def test_pp_micro_batch_pipelining_matches_single_device():
    config = _pp_test_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                   dtype=np.int64)
    out_dense = llama_forward(ids, weights, config)
    pp = shard_llama_for_pp(weights, num_ranks=2)
    out_mb = micro_batch_forward_pp(ids, pp, config, num_micro_batches=4)
    assert_close(out_mb, out_dense, np.float32, name="micro_batch_pp")


def test_pp_rejects_non_divisible_layers():
    config = LlamaConfig(
        vocab_size=8, hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2,  # 3 not divisible by 2
        num_key_value_heads=2, head_dim=4,
    )
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    with pytest.raises(ValueError, match="divisible"):
        shard_llama_for_pp(weights, num_ranks=2)
