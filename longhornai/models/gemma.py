"""Gemma preset configs.

Gemma (1/2) reuses the LonghornAI Llama forward with three architectural
deltas captured by ``LlamaConfig`` extensions:

* **GeGLU** instead of SwiGLU (``mlp_activation="gelu"``).
* **Embedding scaling** by ``sqrt(hidden_size)`` after the embedding lookup
  (``embed_scale``).
* **Large vocab** (256k for Gemma-1, 256k for Gemma-2).

PLAN.md §8 M4 deliverable: "Gemma + Phi support". The toy preset is the
shape-and-feature-equivalent config used by tests; the production-shape
preset is the published Gemma-2-2B (or 7B / 9B / 27B — the toy preset
captures the architectural fingerprint).

Gemma-2's alternating local/global attention (sliding window every other
layer) is a Phase-6 sparse-attention feature (PLAN.md §3 Phase 6) and is
out of scope for M4. Without it the model is architecturally correct, just
not memory-optimal at very long context — the right scope split for M4.
"""

from __future__ import annotations

import math

from .llama import LlamaConfig


def gemma_2b_config() -> LlamaConfig:
    """Gemma-2-2B production-shape config."""
    hidden = 2304
    return LlamaConfig(
        vocab_size=256000,
        hidden_size=hidden,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        embed_scale=math.sqrt(hidden),
        mlp_activation="gelu",
    )


def gemma_toy_config() -> LlamaConfig:
    """Dimensionally-scaled Gemma config for tests (preserves embed-scale + GeGLU)."""
    hidden = 64
    return LlamaConfig(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        embed_scale=math.sqrt(hidden),
        mlp_activation="gelu",
    )


__all__ = ["gemma_2b_config", "gemma_toy_config"]
