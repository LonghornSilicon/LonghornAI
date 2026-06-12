"""Quantized Llama forward + prefix cache + speculative decoding tests (M5)."""

import numpy as np
import pytest

from longhornai.models import (
    LlamaConfig,
    init_random_weights,
    llama_forward,
    llama_forward_quantized,
    quantize_weights_int4,
)
from longhornai.runtime import (
    ContinuousBatchingScheduler,
    PrefixCache,
    Request,
    SchedulerConfig,
    SpeculativeDecoder,
    SpeculativeStats,
    greedy_target_decode,
)


def _toy_config(hidden=32, layers=2):
    return LlamaConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=hidden // 4,
    )


# --- Quantized Llama forward ----------------------------------------------

def test_quantized_llama_forward_shape_and_finite():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    weights_q = quantize_weights_int4(weights, group_size=32)
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out = llama_forward_quantized(ids, weights_q, config)
    assert out.shape == (1, 5, config.vocab_size)
    assert out.dtype == weights.embed_tokens.dtype
    assert np.all(np.isfinite(out))


def test_quantized_llama_logits_track_fp32(rng):
    """W4A16 logits should track the FP32 reference within an INT4-bound MSE."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    weights_q = quantize_weights_int4(weights, group_size=32)
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out_fp = llama_forward(ids, weights, config)
    out_q = llama_forward_quantized(ids, weights_q, config)
    # Per-element quant-induced noise on a 2-layer toy is small; exit-gate
    # bound is per-model perplexity. We use a logit-MSE proxy here.
    mse = float(np.mean((out_fp - out_q) ** 2))
    assert mse < 0.05, f"W4A16 logit MSE = {mse}"


def test_quantize_weights_int4_does_not_quantize_norms_or_embed():
    """Norm + embed tensors stay FP — quantizing them costs accuracy with no perf gain."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    weights_q = quantize_weights_int4(weights, group_size=32)
    assert weights_q.embed_tokens.dtype == np.float32
    assert weights_q.final_norm.dtype == np.float32
    for layer_q in weights_q.layers:
        assert layer_q.input_norm.dtype == np.float32
        assert layer_q.post_attn_norm.dtype == np.float32


# --- Prefix cache ---------------------------------------------------------

def test_prefix_cache_lookup_and_release():
    pc = PrefixCache(block_size=4)
    k = PrefixCache.chain(PrefixCache.root_key(), [1, 2, 3, 4])
    assert pc.acquire(k) is None
    pc.insert(k, block_id=42)
    assert pc.size == 1
    # Acquire bumps refcount; release decrements.
    assert pc.acquire(k) == 42
    assert pc.release(k) is None  # still refcount 1
    freed = pc.release(k)
    assert freed == 42
    assert pc.size == 0


def test_prefix_cache_chain_uniqueness():
    """Different parents → different keys, even with the same suffix."""
    p1 = PrefixCache.chain(PrefixCache.root_key(), [1, 2, 3, 4])
    p2 = PrefixCache.chain(PrefixCache.root_key(), [5, 6, 7, 8])
    s1 = PrefixCache.chain(p1, [9, 10, 11, 12])
    s2 = PrefixCache.chain(p2, [9, 10, 11, 12])
    assert s1 != s2


def test_scheduler_prefix_reuse_two_identical_prompts():
    """When two requests have the same prompt, the second reuses cached blocks."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=4, num_blocks=16, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32,
                        enable_prefix_cache=True),
    )
    # Long-lived requests so the cache stays populated across iterations.
    sched.add_request(Request("r1", prompt.copy(), max_new_tokens=5))
    sched.step()  # admit + prefill r1; cache populated
    initial_cache_size = sched.prefix_cache.size
    assert initial_cache_size > 0

    sched.add_request(Request("r2", prompt.copy(), max_new_tokens=5))
    sched.step()  # admit r2 — should hit the cache

    assert sched.stats.prefix_blocks_reused >= 1
    sched.run_until_done()
    # Both requests must have produced identical outputs (deterministic greedy).
    by_id = {r.request_id: r for r in sched.finished}
    assert by_id["r1"].output_ids == by_id["r2"].output_ids


def test_scheduler_prefix_reuse_matches_greedy_dense():
    """Prefix-cached output must equal the dense-forward greedy output."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

    def greedy(prompt, max_new):
        seq = prompt.copy()
        out = []
        for _ in range(max_new):
            logits = llama_forward(seq.reshape(1, -1), weights, config)
            tok = int(np.argmax(logits[0, -1]))
            out.append(tok)
            seq = np.concatenate([seq, np.array([tok], dtype=seq.dtype)])
        return out
    expected = greedy(prompt, 3)

    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=2, num_blocks=16, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32,
                        enable_prefix_cache=True),
    )
    sched.add_request(Request("r1", prompt.copy(), max_new_tokens=3))
    sched.step()
    sched.add_request(Request("r2", prompt.copy(), max_new_tokens=3))
    sched.run_until_done()

    for r in sched.finished:
        assert r.output_ids == expected


def test_scheduler_prefix_cache_disabled_by_default():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=2, num_blocks=8, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32),
    )
    assert sched.prefix_cache is None


def test_scheduler_prefix_cache_releases_blocks_on_completion():
    """Reused blocks must return to the free pool when the last referrer finishes."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=2, num_blocks=8, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32,
                        enable_prefix_cache=True),
    )
    initial_free = len(sched.free_blocks)
    sched.add_request(Request("r1", prompt.copy(), max_new_tokens=1))
    sched.add_request(Request("r2", prompt.copy(), max_new_tokens=1))
    sched.run_until_done()
    # All blocks back to free; cache empty.
    assert len(sched.free_blocks) == initial_free
    assert sched.prefix_cache.size == 0


# --- Speculative decoding -------------------------------------------------

def test_speculative_decode_lossless_with_oracle_draft():
    """Same draft + target → 100% acceptance, output matches greedy reference."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    spec = SpeculativeDecoder(weights, config, weights, config, speculate_k=4)
    stats = SpeculativeStats()
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    out_spec = spec.generate(prompt, max_new_tokens=8, stats=stats)
    out_ref = greedy_target_decode(prompt, weights, config, max_new_tokens=8)
    assert out_spec == out_ref
    assert stats.acceptance_rate == 1.0
    # 8 tokens in fewer than 8 target calls — that's the speedup.
    assert stats.target_calls < 8


def test_speculative_decode_lossless_with_unaligned_draft():
    """Even with a poor draft (different model), output equals greedy target."""
    target_config = _toy_config(hidden=32, layers=2)
    target_weights = init_random_weights(target_config, dtype=np.float32, seed=0)
    draft_config = _toy_config(hidden=16, layers=1)
    draft_weights = init_random_weights(draft_config, dtype=np.float32, seed=42)
    spec = SpeculativeDecoder(
        target_weights, target_config, draft_weights, draft_config,
        speculate_k=3,
    )
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    out_spec = spec.generate(prompt, max_new_tokens=6)
    out_ref = greedy_target_decode(prompt, target_weights, target_config, max_new_tokens=6)
    assert out_spec == out_ref


def test_speculative_decode_rejects_draft_mismatch_at_first_position():
    """Draft → target mismatch at position 0 should still produce a target-quality token."""
    target_config = _toy_config()
    weights_target = init_random_weights(target_config, dtype=np.float32, seed=0)
    # Use a different-seeded draft that probably disagrees on the very first token.
    weights_draft = init_random_weights(target_config, dtype=np.float32, seed=99)
    spec = SpeculativeDecoder(weights_target, target_config,
                                 weights_draft, target_config, speculate_k=3)
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    out_spec = spec.generate(prompt, max_new_tokens=4)
    out_ref = greedy_target_decode(prompt, weights_target, target_config, max_new_tokens=4)
    assert out_spec == out_ref


def test_speculative_stats_tracks_proposals_and_acceptances():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    spec = SpeculativeDecoder(weights, config, weights, config, speculate_k=2)
    stats = SpeculativeStats()
    prompt = np.array([1, 2], dtype=np.int64)
    spec.generate(prompt, max_new_tokens=4, stats=stats)
    assert stats.proposed >= stats.accepted
    assert stats.target_calls >= 1
