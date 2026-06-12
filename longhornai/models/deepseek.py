"""DeepSeek-V2-style model — Multi-head Latent Attention + fine-grained MoE.

PLAN.md §7 DeepSeek entry / §8 M7 deliverable: "DeepSeek-MoE support".
DeepSeek's two architectural inventions:

* **Multi-head Latent Attention (MLA).** KV state is compressed via a
  low-rank projection — instead of storing per-head ``(n_heads, head_dim)``
  KV per token, the cache holds a small compressed tensor of dim
  ``kv_lora_rank`` plus a separate RoPE component. Decompresses to per-head
  on the read side. Cuts KV bandwidth ~10× at long context. Q is also
  split into a "no positional" part and a "RoPE" part — only the RoPE part
  has rotary applied; the no-pos part is plain.

* **Fine-grained MoE with a shared expert.** Many small routed experts
  (DeepSeek-V2: 160 routed at 1/16 of the dense FFN width) plus one or
  two **always-active shared experts**. Top-K of the routed experts +
  the shared expert outputs are summed per token. The shared expert
  captures common features so the routed experts can specialize.

This module ships the reference forward and toy + production-shape
configs. All math goes through registered LonghornAI kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from ..kernels import (
    embedding_lookup,
    gemm,
    moe_combine,
    moe_dispatch,
    moe_router,
    moe_top_k,
    rmsnorm,
    rope,
    sdpa,
    silu,
)


# --- config + weights --------------------------------------------------------

@dataclass(frozen=True)
class DeepSeekConfig:
    """DeepSeek-V2 hyperparameters.

    MLA parameters:
    * ``q_lora_rank`` / ``kv_lora_rank`` — the compressed dims for Q and KV.
    * ``qk_nope_head_dim`` — per-head no-positional Q/K dim.
    * ``qk_rope_head_dim`` — per-head RoPE Q/K dim. Total per-head QK dim is
      ``qk_nope_head_dim + qk_rope_head_dim``.
    * ``v_head_dim``      — per-head V dim (typically equals qk_nope_head_dim).

    MoE parameters:
    * ``num_routed_experts`` — gated experts.
    * ``num_shared_experts`` — always-active experts.
    * ``num_experts_per_tok`` — top-K over the routed experts.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int            # per-expert intermediate dim
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_routed_experts: int
    num_shared_experts: int
    num_experts_per_tok: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_experts_per_tok > self.num_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) > "
                f"num_routed_experts ({self.num_routed_experts})"
            )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class DeepSeekExpertWeights:
    """One MoE expert's SwiGLU MLP."""

    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass
class DeepSeekLayerWeights:
    """Per-layer weights — MLA attention + fine-grained MoE FFN."""

    input_norm: np.ndarray            # (hidden,)

    # MLA projections — Q is x → q_lora → q_b; KV is x → (kv_lora + rope_k).
    q_a_proj: np.ndarray              # (hidden, q_lora_rank)
    q_a_norm: np.ndarray              # (q_lora_rank,)
    q_b_proj: np.ndarray              # (q_lora_rank, n_heads * qk_head_dim)
    kv_a_proj: np.ndarray             # (hidden, kv_lora_rank + qk_rope_head_dim)
    kv_a_norm: np.ndarray             # (kv_lora_rank,)
    kv_b_proj: np.ndarray             # (kv_lora_rank, n_heads * (qk_nope + v_head))
    o_proj: np.ndarray                # (n_heads * v_head_dim, hidden)

    post_attn_norm: np.ndarray        # (hidden,)

    # Fine-grained MoE: router + routed experts + shared experts.
    router_gate: np.ndarray           # (hidden, num_routed_experts)
    routed_experts: List[DeepSeekExpertWeights]
    shared_experts: List[DeepSeekExpertWeights]


@dataclass
class DeepSeekWeights:
    embed_tokens: np.ndarray
    layers: List[DeepSeekLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray


def init_deepseek_weights(
    config: DeepSeekConfig, *, dtype=np.float32, scale: float = 0.02, seed: int = 0,
) -> DeepSeekWeights:
    """Allocate small random DeepSeek weights for tests."""
    rng = np.random.default_rng(seed)

    def randn(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(dtype)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    n_h = config.num_attention_heads
    H = config.hidden_size
    I = config.intermediate_size

    layers: List[DeepSeekLayerWeights] = []
    for _ in range(config.num_hidden_layers):
        routed = [
            DeepSeekExpertWeights(
                gate_proj=randn(H, I),
                up_proj=randn(H, I),
                down_proj=randn(I, H),
            )
            for _ in range(config.num_routed_experts)
        ]
        shared = [
            DeepSeekExpertWeights(
                gate_proj=randn(H, I),
                up_proj=randn(H, I),
                down_proj=randn(I, H),
            )
            for _ in range(config.num_shared_experts)
        ]
        layers.append(DeepSeekLayerWeights(
            input_norm=ones(H),
            q_a_proj=randn(H, config.q_lora_rank),
            q_a_norm=ones(config.q_lora_rank),
            q_b_proj=randn(config.q_lora_rank, n_h * config.qk_head_dim),
            kv_a_proj=randn(H, config.kv_lora_rank + config.qk_rope_head_dim),
            kv_a_norm=ones(config.kv_lora_rank),
            kv_b_proj=randn(config.kv_lora_rank,
                            n_h * (config.qk_nope_head_dim + config.v_head_dim)),
            o_proj=randn(n_h * config.v_head_dim, H),
            post_attn_norm=ones(H),
            router_gate=randn(H, config.num_routed_experts),
            routed_experts=routed,
            shared_experts=shared,
        ))
    return DeepSeekWeights(
        embed_tokens=randn(config.vocab_size, H),
        layers=layers,
        final_norm=ones(H),
        lm_head=randn(H, config.vocab_size),
    )


# --- MLA attention -----------------------------------------------------------

def _mla_attention(x, w: DeepSeekLayerWeights, config: DeepSeekConfig, attn_impl):
    """Multi-head Latent Attention — low-rank Q + KV with split RoPE / no-pos parts."""
    B, S, H = x.shape
    n_h = config.num_attention_heads
    nope_d = config.qk_nope_head_dim
    rope_d = config.qk_rope_head_dim
    v_d = config.v_head_dim
    qk_d = config.qk_head_dim
    dt = x.dtype
    x_flat = x.reshape(B * S, H)

    # --- Q path: x → q_a → norm → q_b → split into no-pos + RoPE.
    q_a = gemm(x_flat, w.q_a_proj)                                  # (B*S, q_lora)
    q_a = rmsnorm(q_a.reshape(B, S, -1), weight=w.q_a_norm,
                  eps=config.rms_norm_eps).reshape(B * S, -1)
    q_full = gemm(q_a, w.q_b_proj).reshape(B, S, n_h, qk_d).transpose(0, 2, 1, 3)
    q_nope = q_full[..., :nope_d]                                   # (B, n_h, S, nope_d)
    q_rope = q_full[..., nope_d:]                                   # (B, n_h, S, rope_d)
    q_rope = rope(q_rope, theta=config.rope_theta)

    # --- KV path: x → kv_a → split into compressed + RoPE-K.
    kv_a_full = gemm(x_flat, w.kv_a_proj).reshape(B, S, -1)         # (B, S, kv_lora + rope_d)
    kv_a = kv_a_full[..., : config.kv_lora_rank]                    # (B, S, kv_lora)
    k_rope_in = kv_a_full[..., config.kv_lora_rank:]                # (B, S, rope_d)
    kv_a_normed = rmsnorm(kv_a, weight=w.kv_a_norm,
                          eps=config.rms_norm_eps).reshape(B * S, -1)
    kv_b = gemm(kv_a_normed, w.kv_b_proj).reshape(
        B, S, n_h, nope_d + v_d,
    ).transpose(0, 2, 1, 3)                                          # (B, n_h, S, nope+v)
    k_nope = kv_b[..., :nope_d]                                      # (B, n_h, S, nope_d)
    v = kv_b[..., nope_d:]                                           # (B, n_h, S, v_d)

    # K's RoPE part is *shared across heads* — broadcast it.
    k_rope = k_rope_in.reshape(B, 1, S, rope_d)
    k_rope = rope(k_rope, theta=config.rope_theta)
    k_rope = np.broadcast_to(k_rope, (B, n_h, S, rope_d)).copy()

    # Concat nope + rope to form the per-head Q and K vectors.
    q = np.concatenate([q_nope, q_rope], axis=-1)                    # (B, n_h, S, qk_d)
    k = np.concatenate([k_nope, k_rope], axis=-1)                    # (B, n_h, S, qk_d)

    # SDPA. Note: V's head_dim differs from QK's; the kernel handles this
    # correctly because v participates only in the final p @ v matmul.
    out = attn_impl(q, k, v, causal=True)                            # (B, n_h, S, v_d)
    out = out.transpose(0, 2, 1, 3).reshape(B * S, n_h * v_d).astype(dt)
    return gemm(out, w.o_proj).reshape(B, S, H)


# --- fine-grained MoE FFN ---------------------------------------------------

def _expert_swiglu(x_for_expert, expert: DeepSeekExpertWeights):
    if x_for_expert.shape[0] == 0:
        H = expert.down_proj.shape[1]
        return np.zeros((0, H), dtype=x_for_expert.dtype)
    g = silu(gemm(x_for_expert, expert.gate_proj))
    u = gemm(x_for_expert, expert.up_proj)
    fused = (g.astype(np.float32) * u.astype(np.float32)).astype(x_for_expert.dtype)
    return gemm(fused, expert.down_proj)


def _deepseek_moe_ffn(x, w: DeepSeekLayerWeights, config: DeepSeekConfig):
    """Routed top-K MoE plus always-active shared experts."""
    B, S, H = x.shape
    tokens = B * S
    x_flat = x.reshape(tokens, H)

    # --- Routed path.
    logits = moe_router(x_flat, w.router_gate)
    expert_ids, weights = moe_top_k(
        logits, k=config.num_experts_per_tok, normalize=True,
    )
    permuted, offsets, recovery = moe_dispatch(
        x_flat, expert_ids, num_experts=config.num_routed_experts,
    )
    expert_outs = np.empty_like(permuted)
    for e in range(config.num_routed_experts):
        s, t = int(offsets[e]), int(offsets[e + 1])
        if t > s:
            expert_outs[s:t] = _expert_swiglu(permuted[s:t], w.routed_experts[e])
    routed_out = moe_combine(
        expert_outs, weights, recovery, n_tokens=tokens, hidden=H,
    )

    # --- Shared path: every token is routed to every shared expert with
    # weight 1.0. We just sum their outputs per token.
    shared_out = np.zeros_like(routed_out)
    for shared in w.shared_experts:
        shared_out = shared_out + _expert_swiglu(x_flat, shared)

    return (routed_out + shared_out).reshape(B, S, H)


# --- decoder layer + forward ------------------------------------------------

AttnImpl = Callable[..., np.ndarray]


def deepseek_decoder_layer(
    x: np.ndarray, w: DeepSeekLayerWeights, config: DeepSeekConfig, *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """One DeepSeek decoder block: pre-norm MLA + pre-norm fine-grained MoE."""
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    x = x + _mla_attention(h, w, config, attn_impl)
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _deepseek_moe_ffn(h, w, config)
    return x


def deepseek_forward(
    input_ids: np.ndarray,
    weights: DeepSeekWeights,
    config: DeepSeekConfig,
    *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Full DeepSeek-V2 forward."""
    x = embedding_lookup(weights.embed_tokens, input_ids)
    for layer in weights.layers:
        x = deepseek_decoder_layer(x, layer, config, attn_impl=attn_impl)
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    return gemm(x.reshape(B * S, H), weights.lm_head).reshape(B, S, config.vocab_size)


# --- presets -----------------------------------------------------------------

def deepseek_v2_config() -> DeepSeekConfig:
    """DeepSeek-V2 production-shape config (236B total / 21B active)."""
    return DeepSeekConfig(
        vocab_size=102400,
        hidden_size=5120,
        intermediate_size=1536,
        num_hidden_layers=60,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_routed_experts=160,
        num_shared_experts=2,
        num_experts_per_tok=6,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def deepseek_toy_config() -> DeepSeekConfig:
    """Dimensionally-scaled DeepSeek config for tests.

    Captures the architectural fingerprint: low-rank Q + KV, split RoPE / no-pos
    head dims, fine-grained routed MoE + 1 shared expert.
    """
    return DeepSeekConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


__all__ = [
    "DeepSeekConfig",
    "DeepSeekExpertWeights",
    "DeepSeekLayerWeights",
    "DeepSeekWeights",
    "init_deepseek_weights",
    "deepseek_decoder_layer",
    "deepseek_forward",
    "deepseek_v2_config",
    "deepseek_toy_config",
]
