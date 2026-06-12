"""Qwen preset configs.

Qwen-2 / Qwen-2.5 share the same pre-norm RMSNorm + GQA + SwiGLU + RoPE
architecture as Llama-3 — the distinguishing detail is the **RoPE base**
(``rope_theta = 1_000_000.0`` instead of 10_000.0), which extends the
representable position range proportionally. The forward function is the
same one used for Llama; the config is what changes.

PLAN.md §8 M3 deliverable: "Mistral + Qwen FP16 correct".
"""

from __future__ import annotations

from .llama import LlamaConfig


def qwen2_5_7b_config() -> LlamaConfig:
    """Qwen-2.5-7B production-shape config."""
    return LlamaConfig(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
    )


def qwen_toy_config() -> LlamaConfig:
    """Dimensionally-scaled Qwen config for tests (7:1 GQA + Qwen's RoPE base)."""
    return LlamaConfig(
        vocab_size=128,
        hidden_size=56,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
    )


__all__ = ["qwen2_5_7b_config", "qwen_toy_config"]
