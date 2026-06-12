"""Mixtral model — Llama-style attention + Sparse-MoE FFN.

PLAN.md §7 Mixtral entry / §8 M6 exit gate component "Mixtral decode
correct". Mixtral keeps Llama's attention path (RMSNorm + GQA + RoPE) and
swaps the dense SwiGLU MLP for a Top-K-of-N sparse-MoE block where each
expert is itself a SwiGLU MLP.

The MoE block:

1. ``router(x) = x @ W_gate``  →  ``(tokens, num_experts)`` logits.
2. ``top_k`` selects the ``k`` highest-scoring experts per token and
   softmax-normalizes the selected logits.
3. ``dispatch`` permutes tokens so each expert sees a contiguous slab.
4. Each expert runs SwiGLU(``x @ W_gate_e``) * (``x @ W_up_e``) → ``x @ W_down_e``.
5. ``combine`` un-permutes and weighted-sums the per-expert outputs back
   to per-token form.

The toy preset is dimensionally scaled (4 experts, top-2) so tests run
quickly; the production preset captures the published Mixtral-8x7B numbers.

Expert parallelism — sharding experts across virtual ranks — lives in
:mod:`longhornai.runtime.expert_parallel` and runs the same ``mixtral_forward``
math under an all-to-all token routing pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from ..kernels import (
    embedding_lookup,
    gemm,
    grouped_gemm,
    moe_combine,
    moe_dispatch,
    moe_router,
    moe_top_k,
    rmsnorm,
    rope,
    sdpa,
    silu,
)
from .llama import (
    AttnImpl,
    LlamaConfig,
    LlamaWeights,
    _attention_block,
    init_random_weights as _init_llama_weights,
)


# --- config + weights --------------------------------------------------------

@dataclass(frozen=True)
class MixtralConfig:
    """Mixtral hyperparameters — Llama base + MoE FFN parameters."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_local_experts: int
    num_experts_per_tok: int            # top-K
    rope_theta: float = 1_000_000.0     # Mixtral's distinctive RoPE base
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads"
            )
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) > "
                f"num_local_experts ({self.num_local_experts})"
            )

    def to_llama(self) -> LlamaConfig:
        """Project to a LlamaConfig for the attention path's bookkeeping."""
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )


@dataclass
class MixtralExpertWeights:
    """One expert's SwiGLU MLP weights."""

    gate_proj: np.ndarray   # (hidden, intermediate)
    up_proj: np.ndarray     # (hidden, intermediate)
    down_proj: np.ndarray   # (intermediate, hidden)


@dataclass
class MixtralLayerWeights:
    """Per-layer weights — Llama-style attention + sparse-MoE FFN."""

    input_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    post_attn_norm: np.ndarray
    router_gate: np.ndarray                       # (hidden, num_local_experts)
    experts: List[MixtralExpertWeights]


@dataclass
class MixtralWeights:
    embed_tokens: np.ndarray
    layers: List[MixtralLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray


def init_mixtral_weights(
    config: MixtralConfig, *, dtype=np.float32, scale: float = 0.02, seed: int = 0,
) -> MixtralWeights:
    """Allocate small random Mixtral weights for tests — not a training init."""
    rng = np.random.default_rng(seed)

    def randn(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(dtype)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    H = config.hidden_size
    I = config.intermediate_size
    D = config.head_dim
    E = config.num_local_experts

    layers = [
        MixtralLayerWeights(
            input_norm=ones(H),
            q_proj=randn(H, n_q * D),
            k_proj=randn(H, n_kv * D),
            v_proj=randn(H, n_kv * D),
            o_proj=randn(n_q * D, H),
            post_attn_norm=ones(H),
            router_gate=randn(H, E),
            experts=[
                MixtralExpertWeights(
                    gate_proj=randn(H, I),
                    up_proj=randn(H, I),
                    down_proj=randn(I, H),
                )
                for _ in range(E)
            ],
        )
        for _ in range(config.num_hidden_layers)
    ]
    return MixtralWeights(
        embed_tokens=randn(config.vocab_size, H),
        layers=layers,
        final_norm=ones(H),
        lm_head=randn(H, config.vocab_size),
    )


# --- Sparse-MoE FFN ----------------------------------------------------------

def _expert_swiglu(x_for_expert: np.ndarray, expert: MixtralExpertWeights) -> np.ndarray:
    """One expert's SwiGLU MLP on its share of tokens."""
    if x_for_expert.shape[0] == 0:
        # No tokens routed to this expert — return a zero output of correct shape.
        H = expert.down_proj.shape[1]
        return np.zeros((0, H), dtype=x_for_expert.dtype)
    gate = silu(gemm(x_for_expert, expert.gate_proj))
    up = gemm(x_for_expert, expert.up_proj)
    fused = (gate.astype(np.float32) * up.astype(np.float32)).astype(x_for_expert.dtype)
    return gemm(fused, expert.down_proj)


def _moe_ffn(
    x: np.ndarray, w: MixtralLayerWeights, config: MixtralConfig,
) -> np.ndarray:
    """Sparse-MoE FFN: route → per-expert SwiGLU → combine."""
    B, S, H = x.shape
    tokens = B * S
    x_flat = x.reshape(tokens, H)

    logits = moe_router(x_flat, w.router_gate)
    expert_ids, weights = moe_top_k(
        logits, k=config.num_experts_per_tok, normalize=True,
    )
    permuted, offsets, recovery = moe_dispatch(
        x_flat, expert_ids, num_experts=config.num_local_experts,
    )

    # Per-expert SwiGLU. Backends with grouped-GEMM hardware would issue
    # one batched call; the CPU reference loops for clarity.
    expert_outs = np.empty_like(permuted)
    for e in range(config.num_local_experts):
        start, end = int(offsets[e]), int(offsets[e + 1])
        if end > start:
            expert_outs[start:end] = _expert_swiglu(permuted[start:end], w.experts[e])

    combined = moe_combine(
        expert_outs, weights, recovery, n_tokens=tokens, hidden=H,
    )
    return combined.reshape(B, S, H)


def mixtral_decoder_layer(
    x: np.ndarray, w: MixtralLayerWeights, config: MixtralConfig, *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """One Mixtral decoder block: pre-norm attention + pre-norm MoE FFN."""
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    # The attention block is identical to Llama's; we reuse the helper by
    # giving it a LlamaLayerWeights view with the current layer's projections.
    from .llama import LlamaLayerWeights
    attn_view = LlamaLayerWeights(
        input_norm=w.input_norm,
        q_proj=w.q_proj, k_proj=w.k_proj, v_proj=w.v_proj, o_proj=w.o_proj,
        post_attn_norm=w.post_attn_norm,
        # MoE has no dense gate/up/down — fill with placeholders that the
        # attention helper never touches.
        gate_proj=np.zeros((1, 1), dtype=x.dtype),
        up_proj=np.zeros((1, 1), dtype=x.dtype),
        down_proj=np.zeros((1, 1), dtype=x.dtype),
    )
    x = x + _attention_block(h, attn_view, config.to_llama(), attn_impl)
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _moe_ffn(h, w, config)
    return x


def mixtral_forward(
    input_ids: np.ndarray,
    weights: MixtralWeights,
    config: MixtralConfig,
    *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Full Mixtral forward. Returns logits ``(B, S, vocab_size)``."""
    x = embedding_lookup(weights.embed_tokens, input_ids)
    for layer in weights.layers:
        x = mixtral_decoder_layer(x, layer, config, attn_impl=attn_impl)
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    return gemm(x.reshape(B * S, H), weights.lm_head).reshape(B, S, config.vocab_size)


# --- presets -----------------------------------------------------------------

def mixtral_8x7b_config() -> MixtralConfig:
    """Production-shape Mixtral-8x7B."""
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_local_experts=8,
        num_experts_per_tok=2,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-5,
    )


def mixtral_toy_config() -> MixtralConfig:
    """Dimensionally-scaled Mixtral config for tests (4 experts, top-2)."""
    return MixtralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-5,
    )


__all__ = [
    "MixtralConfig",
    "MixtralExpertWeights",
    "MixtralLayerWeights",
    "MixtralWeights",
    "init_mixtral_weights",
    "mixtral_decoder_layer",
    "mixtral_forward",
    "mixtral_8x7b_config",
    "mixtral_toy_config",
]
