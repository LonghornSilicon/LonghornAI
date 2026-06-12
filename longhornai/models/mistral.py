"""Mistral preset configs.

Mistral-7B is architecturally a Llama-3-style decoder: pre-norm RMSNorm +
GQA + SwiGLU + RoPE, all running through the same forward function as Llama
(:func:`longhornai.models.llama_forward`). The distinguishing details are
GQA ratio (32 Q heads, 8 KV heads — 4:1) and the integer constants.

PLAN.md §8 M3 deliverable: "Mistral + Qwen FP16 correct". The :func:`mistral_7b_config`
factory below returns a real-shaped config; :func:`mistral_toy_config` is the
dimensionally-scaled version used in tests.

Sliding-window attention — Mistral's distinctive optimization — lands in
Phase 6 (PLAN.md §3 Phase 6 sparse attention). Without SWA the model is
*architecturally correct*, just less memory-efficient at long context;
that's the right scope split for M3.
"""

from __future__ import annotations

from .llama import LlamaConfig


def mistral_7b_config() -> LlamaConfig:
    """Mistral-7B production-shape config."""
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


def mistral_toy_config() -> LlamaConfig:
    """Dimensionally-scaled Mistral config for tests (4:1 GQA preserved)."""
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


__all__ = ["mistral_7b_config", "mistral_toy_config"]
