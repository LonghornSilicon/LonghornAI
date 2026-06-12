"""Gemma + Phi forward tests (M4)."""

import math

import numpy as np
import pytest

from longhornai.models import (
    LlamaConfig,
    gemma_2b_config,
    gemma_toy_config,
    init_random_weights,
    llama_forward,
    phi3_medium_config,
    phi3_mini_config,
    phi_toy_config,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_gemma_forward_finite(dtype):
    config = gemma_toy_config()
    weights = init_random_weights(config, dtype=dtype, seed=0, scale=0.01)
    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out = llama_forward(input_ids, weights, config)
    assert out.shape == (1, 5, config.vocab_size)
    assert out.dtype == dtype
    assert np.all(np.isfinite(out))


def test_gemma_uses_geglu_and_embed_scale():
    c = gemma_toy_config()
    assert c.mlp_activation == "gelu"
    assert c.embed_scale == math.sqrt(c.hidden_size)


def test_gemma_2b_real_dimensions():
    c = gemma_2b_config()
    assert c.hidden_size == 2304
    assert c.num_attention_heads == 8
    assert c.num_key_value_heads == 4
    assert c.embed_scale == math.sqrt(c.hidden_size)
    assert c.mlp_activation == "gelu"


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_phi_forward_finite(dtype):
    config = phi_toy_config()
    weights = init_random_weights(config, dtype=dtype, seed=0)
    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out = llama_forward(input_ids, weights, config)
    assert out.shape == (1, 5, config.vocab_size)
    assert out.dtype == dtype
    assert np.all(np.isfinite(out))


def test_phi_uses_gelu():
    assert phi_toy_config().mlp_activation == "gelu"


def test_phi3_mini_real_dimensions():
    c = phi3_mini_config()
    assert c.hidden_size == 3072
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 32  # MHA
    assert c.mlp_activation == "gelu"


def test_phi3_medium_real_dimensions():
    c = phi3_medium_config()
    assert c.hidden_size == 5120
    assert c.num_attention_heads == 40
    assert c.num_key_value_heads == 10  # GQA 4:1
    assert c.mlp_activation == "gelu"


def test_embed_scale_changes_forward_output():
    """If embed_scale is dead, scaled vs unscaled outputs would match."""
    base = phi_toy_config()
    scaled = LlamaConfig(
        vocab_size=base.vocab_size,
        hidden_size=base.hidden_size,
        intermediate_size=base.intermediate_size,
        num_hidden_layers=base.num_hidden_layers,
        num_attention_heads=base.num_attention_heads,
        num_key_value_heads=base.num_key_value_heads,
        head_dim=base.head_dim,
        rope_theta=base.rope_theta,
        rms_norm_eps=base.rms_norm_eps,
        embed_scale=4.0,
        mlp_activation=base.mlp_activation,
    )
    weights = init_random_weights(base, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3]], dtype=np.int64)
    out1 = llama_forward(ids, weights, base)
    out2 = llama_forward(ids, weights, scaled)
    assert not np.allclose(out1, out2)


def test_mlp_activation_changes_forward_output():
    base_cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
        mlp_activation="silu",
    )
    gelu_cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
        mlp_activation="gelu",
    )
    weights = init_random_weights(base_cfg, dtype=np.float32, seed=0)
    ids = np.array([[1, 2, 3]], dtype=np.int64)
    out_silu = llama_forward(ids, weights, base_cfg)
    out_gelu = llama_forward(ids, weights, gelu_cfg)
    assert not np.allclose(out_silu, out_gelu)


def test_invalid_mlp_activation_rejected():
    with pytest.raises(ValueError, match="mlp_activation"):
        LlamaConfig(
            vocab_size=8, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=4,
            mlp_activation="relu",
        )
