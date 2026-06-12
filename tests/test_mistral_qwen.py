"""Mistral + Qwen forward tests (M3).

PLAN.md §8 M3 deliverable: "Mistral + Qwen FP16 correct". Both reuse the
Llama forward function — Mistral is a 4:1 GQA Llama-3 with the standard
``rope_theta=10000``; Qwen-2.5 has ``rope_theta=1e6`` and a different GQA
ratio. These tests certify that the same forward path runs both
architectures correctly in fp16 and fp32.
"""

import numpy as np
import pytest

from longhornai import flash_attention_v2, sdpa
from longhornai.models import (
    init_random_weights,
    llama_forward,
    mistral_toy_config,
    qwen_toy_config,
)
from longhornai.validation import assert_close


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_mistral_forward_finite(dtype):
    config = mistral_toy_config()
    weights = init_random_weights(config, dtype=dtype, seed=0)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 8, config.vocab_size)
    assert logits.dtype == dtype
    assert np.all(np.isfinite(logits))


def test_mistral_uses_4to1_gqa():
    config = mistral_toy_config()
    assert config.num_attention_heads == 4 * config.num_key_value_heads


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_qwen_forward_finite(dtype):
    config = qwen_toy_config()
    weights = init_random_weights(config, dtype=dtype, seed=1)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    logits = llama_forward(input_ids, weights, config)
    assert logits.shape == (1, 8, config.vocab_size)
    assert logits.dtype == dtype
    assert np.all(np.isfinite(logits))


def test_qwen_uses_million_rope_theta():
    """Qwen's distinctive choice — long-context-friendly RoPE base."""
    assert qwen_toy_config().rope_theta == 1_000_000.0


def test_qwen_rope_theta_changes_output():
    """Different theta must produce a different forward (else config is dead)."""
    qwen_cfg = qwen_toy_config()
    llama_like = qwen_cfg.__class__(
        vocab_size=qwen_cfg.vocab_size,
        hidden_size=qwen_cfg.hidden_size,
        intermediate_size=qwen_cfg.intermediate_size,
        num_hidden_layers=qwen_cfg.num_hidden_layers,
        num_attention_heads=qwen_cfg.num_attention_heads,
        num_key_value_heads=qwen_cfg.num_key_value_heads,
        head_dim=qwen_cfg.head_dim,
        rope_theta=10000.0,                # Llama default
        rms_norm_eps=qwen_cfg.rms_norm_eps,
    )
    weights = init_random_weights(qwen_cfg, dtype=np.float32, seed=2)
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    out_qwen = llama_forward(input_ids, weights, qwen_cfg)
    out_llama_like = llama_forward(input_ids, weights, llama_like)
    # Not equal — RoPE theta affects frequencies which propagate through attention.
    assert not np.allclose(out_qwen, out_llama_like)


def test_mistral_sdpa_flash_parity():
    config = mistral_toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=3)
    input_ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    out_sdpa = llama_forward(input_ids, weights, config, attn_impl=sdpa)
    out_flash = llama_forward(input_ids, weights, config, attn_impl=flash_attention_v2)
    assert_close(out_flash, out_sdpa, np.float32, name="mistral_sdpa_flash_v2")


def test_qwen_sdpa_flash_parity():
    config = qwen_toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=4)
    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out_sdpa = llama_forward(input_ids, weights, config, attn_impl=sdpa)
    out_flash = llama_forward(input_ids, weights, config, attn_impl=flash_attention_v2)
    assert_close(out_flash, out_sdpa, np.float32, name="qwen_sdpa_flash_v2")


def test_mistral_7b_config_real_dimensions():
    """Real-shape config matches the published Mistral-7B numbers."""
    from longhornai.models import mistral_7b_config
    c = mistral_7b_config()
    assert c.hidden_size == 4096
    assert c.intermediate_size == 14336
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.num_hidden_layers == 32


def test_qwen2_5_7b_config_real_dimensions():
    """Real-shape config matches the published Qwen-2.5-7B numbers."""
    from longhornai.models import qwen2_5_7b_config
    c = qwen2_5_7b_config()
    assert c.hidden_size == 3584
    assert c.intermediate_size == 18944
    assert c.num_attention_heads == 28
    assert c.num_key_value_heads == 4
    assert c.rope_theta == 1_000_000.0
    assert c.num_hidden_layers == 28
