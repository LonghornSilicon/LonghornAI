"""Llama forward tests (M2).

PLAN.md §8 M2 exit-gate item: "Llama FP16 forward correct (CPU/sim)". The
forward — embedding → N decoder layers → final norm → LM head — runs through
registered LonghornAI kernels. These tests certify shape and dtype contracts,
finiteness in fp16 (no overflow / NaN), MHA + GQA paths, and SDPA↔Flash-v1
parity through the entire stack.
"""

import numpy as np
import pytest

from longhornai import flash_attention_v1, sdpa
from longhornai.models import LlamaConfig, init_random_weights, llama_forward
from longhornai.validation import assert_close


def _toy_config(*, n_q=4, n_kv=2, layers=2, vocab=64, hidden=32, inter=64, head_dim=8):
    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
    )


# --- shape + finite + dtype contracts -------------------------------------

def test_llama_fp32_forward_shape_and_finite():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 8, config.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))


def test_llama_fp16_forward_correct():
    """M2 exit gate: Llama FP16 forward is finite and shape-correct."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float16, seed=42, scale=0.02)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 8, config.vocab_size)
    assert logits.dtype == np.float16
    assert np.all(np.isfinite(logits)), "FP16 forward must not overflow / NaN"


def test_llama_batched_forward():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=1)
    input_ids = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 0], [10, 20, 30, 40]], dtype=np.int64
    )
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (3, 4, config.vocab_size)
    assert np.all(np.isfinite(logits))


# --- MHA / GQA paths ------------------------------------------------------

def test_llama_mha_path():
    """num_kv_heads == num_attention_heads → classic MHA, no broadcasting."""
    config = _toy_config(n_q=4, n_kv=4)
    weights = init_random_weights(config, dtype=np.float32, seed=2)
    input_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 4, config.vocab_size)
    assert np.all(np.isfinite(logits))


@pytest.mark.parametrize("n_q,n_kv", [(6, 2), (4, 1), (8, 4)])
def test_llama_gqa_path(n_q, n_kv):
    """num_kv_heads < num_attention_heads → GQA via head replication."""
    config = _toy_config(n_q=n_q, n_kv=n_kv, hidden=n_q * 4, head_dim=4)
    weights = init_random_weights(config, dtype=np.float32, seed=3)
    input_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 4, config.vocab_size)
    assert np.all(np.isfinite(logits))


def test_llama_rejects_non_divisible_head_groups():
    with pytest.raises(ValueError, match="multiple"):
        LlamaConfig(
            vocab_size=8, hidden_size=12, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=5,
            num_key_value_heads=2, head_dim=4,
        )


# --- SDPA ↔ Flash-v1 parity through the full forward ----------------------

def test_llama_sdpa_flash_parity_fp32():
    """End-to-end forward must agree under the two attention impls."""
    config = _toy_config(n_q=4, n_kv=2, layers=2)
    weights = init_random_weights(config, dtype=np.float32, seed=7)
    input_ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_sdpa = llama_forward(input_ids, weights, config, attn_impl=sdpa)
    out_flash = llama_forward(
        input_ids, weights, config, attn_impl=flash_attention_v1
    )
    assert_close(out_flash, out_sdpa, np.float32, name="llama_sdpa_flash_fp32")


def test_llama_sdpa_flash_parity_fp16():
    config = _toy_config(n_q=4, n_kv=2, layers=2)
    weights = init_random_weights(config, dtype=np.float16, seed=8, scale=0.02)
    input_ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_sdpa = llama_forward(input_ids, weights, config, attn_impl=sdpa)
    out_flash = llama_forward(
        input_ids, weights, config, attn_impl=flash_attention_v1
    )
    assert_close(out_flash, out_sdpa, np.float16, name="llama_sdpa_flash_fp16")


# --- single-layer / single-token sanity ----------------------------------

def test_llama_single_layer_single_token():
    config = _toy_config(layers=1)
    weights = init_random_weights(config, dtype=np.float32, seed=9)
    input_ids = np.array([[3]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 1, config.vocab_size)
    assert np.all(np.isfinite(logits))
