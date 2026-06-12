"""Phi preset configs.

PLAN.md §7 / §8 M4 — "Phi: Compact dense decoder". Phi-3 is the canonical
modern Phi: the LonghornAI Llama forward with a smaller GQA fanout and the
GELU MLP path (``mlp_activation="gelu"``). The toy preset captures the
architectural fingerprint — small dense decoder, GQA, GELU MLP — at the
shape scale our reference backend can run quickly.
"""

from __future__ import annotations

from .llama import LlamaConfig


def phi3_mini_config() -> LlamaConfig:
    """Phi-3-mini (~3.8B) production-shape config."""
    return LlamaConfig(
        vocab_size=32064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,        # MHA in Phi-3-mini
        head_dim=96,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        mlp_activation="gelu",
    )


def phi3_medium_config() -> LlamaConfig:
    """Phi-3-medium (~14B) production-shape config (GQA 5:1)."""
    return LlamaConfig(
        vocab_size=32064,
        hidden_size=5120,
        intermediate_size=17920,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=10,
        head_dim=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        mlp_activation="gelu",
    )


def phi_toy_config() -> LlamaConfig:
    """Dimensionally-scaled Phi config for tests (small dense, GELU MLP)."""
    return LlamaConfig(
        vocab_size=128,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=12,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        mlp_activation="gelu",
    )


__all__ = ["phi3_mini_config", "phi3_medium_config", "phi_toy_config"]
