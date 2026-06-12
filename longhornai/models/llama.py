"""Llama-style transformer forward, composed from LonghornAI kernels.

M2 shipped the dense (non-paged) forward used by the FP16-correctness
exit gate. M4 adds:

* :class:`LlamaConfig` extensions — ``embed_scale`` and ``mlp_activation``
  so Gemma (GeGLU + sqrt-hidden embed scaling) and Phi (GELU MLP) reuse the
  same forward.
* :class:`PagedKVState` — per-layer paged KV pool + per-request block table
  + per-request sequence lengths.
* :func:`llama_prefill` — process a multi-token prompt, write KV into the
  paged pool, return logits for the prompt's last token (the one the
  scheduler samples first).
* :func:`llama_decode_step` — process a single new token per request,
  append KV via the paged pool, return next-token logits. Memory-bound;
  the inner loop of continuous batching (PLAN.md §3 Phase 3).

Both consume a :class:`PagedKVState` allocated by the scheduler in
:mod:`longhornai.runtime.scheduler`. The same kernel surface — RMSNorm,
RoPE, GEMM, paged_attention, SiLU/GELU, embedding_lookup — backs all paths,
so cross-target equivalence (PLAN.md §5.2) holds end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..kernels import (
    embedding_lookup,
    gelu,
    gemm,
    paged_attention as _paged_attention,
    paged_kv_alloc,
    paged_kv_append,
    rmsnorm,
    rope,
    sdpa,
    silu,
)


# --- config & weights --------------------------------------------------------

@dataclass(frozen=True)
class LlamaConfig:
    """Llama-architecture hyperparameters.

    ``num_key_value_heads < num_attention_heads`` selects GQA; equality is
    classic multi-head attention.

    ``embed_scale`` (default 1.0) multiplies the post-embedding hidden state
    — Gemma uses ``sqrt(hidden_size)`` here.

    ``mlp_activation`` (default ``"silu"``) selects the gated activation:
    ``"silu"`` → SwiGLU (Llama, Mistral, Qwen), ``"gelu"`` → GeGLU (Gemma,
    Phi MLPs).
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    embed_scale: float = 1.0
    mlp_activation: str = "silu"

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads "
                f"(got {self.num_attention_heads} and {self.num_key_value_heads})"
            )
        if self.mlp_activation not in ("silu", "gelu"):
            raise ValueError(
                f"mlp_activation must be 'silu' or 'gelu', got {self.mlp_activation!r}"
            )


@dataclass
class LlamaLayerWeights:
    input_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    post_attn_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass
class LlamaWeights:
    embed_tokens: np.ndarray
    layers: List[LlamaLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray


def init_random_weights(
    config: LlamaConfig,
    *,
    dtype=np.float32,
    scale: float = 0.02,
    seed: int = 0,
) -> LlamaWeights:
    """Allocate small random weights for tests — not a training init scheme."""
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

    layers = [
        LlamaLayerWeights(
            input_norm=ones(H),
            q_proj=randn(H, n_q * D),
            k_proj=randn(H, n_kv * D),
            v_proj=randn(H, n_kv * D),
            o_proj=randn(n_q * D, H),
            post_attn_norm=ones(H),
            gate_proj=randn(H, I),
            up_proj=randn(H, I),
            down_proj=randn(I, H),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return LlamaWeights(
        embed_tokens=randn(config.vocab_size, H),
        layers=layers,
        final_norm=ones(H),
        lm_head=randn(H, config.vocab_size),
    )


# --- paged KV state ----------------------------------------------------------

@dataclass
class PagedKVState:
    """Per-layer paged KV pool + per-request block table + sequence lengths.

    The pool is shared across layers via a list of arrays (one per layer);
    the block table and ``seq_lens`` are shared across all layers since the
    logical→physical mapping is the same per request.
    """

    block_size: int
    cache_k: List[np.ndarray]    # per-layer (num_blocks, n_kv, block_size, D)
    cache_v: List[np.ndarray]
    block_table: np.ndarray      # (B, max_blocks_per_seq) int32
    seq_lens: np.ndarray         # (B,) int32


def alloc_paged_kv_state(
    config: LlamaConfig,
    *,
    batch_size: int,
    num_blocks: int,
    block_size: int,
    max_blocks_per_seq: int,
    dtype=np.float16,
) -> PagedKVState:
    """Allocate a fresh paged KV state for ``batch_size`` requests."""
    cache_k = []
    cache_v = []
    for _ in range(config.num_hidden_layers):
        ck, cv = paged_kv_alloc(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
            dtype=dtype,
        )
        cache_k.append(ck)
        cache_v.append(cv)
    return PagedKVState(
        block_size=block_size,
        cache_k=cache_k,
        cache_v=cache_v,
        block_table=np.full((batch_size, max_blocks_per_seq), -1, dtype=np.int32),
        seq_lens=np.zeros((batch_size,), dtype=np.int32),
    )


# --- non-paged forward (M2 path, kept for the existing test surface) ---------

AttnImpl = Callable[..., np.ndarray]


def _gated_activation(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "silu":
        return silu(x)
    if kind == "gelu":
        return gelu(x, approximate="tanh")
    raise ValueError(f"unknown mlp_activation '{kind}'")


def _attention_block(x, w, config, attn_impl):
    B, S, H = x.shape
    n_q, n_kv, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    dt = x.dtype
    x_flat = x.reshape(B * S, H)
    q = gemm(x_flat, w.q_proj).reshape(B, S, n_q, D).transpose(0, 2, 1, 3)
    k = gemm(x_flat, w.k_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)
    v = gemm(x_flat, w.v_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)
    q = rope(q, theta=config.rope_theta)
    k = rope(k, theta=config.rope_theta)
    if n_kv != n_q:
        repeats = n_q // n_kv
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)
    out = attn_impl(q, k, v, causal=True)
    out = out.transpose(0, 2, 1, 3).reshape(B * S, n_q * D).astype(dt)
    return gemm(out, w.o_proj).reshape(B, S, H)


def _mlp_block(x, w, config):
    B, S, H = x.shape
    x_flat = x.reshape(B * S, H)
    gate = _gated_activation(gemm(x_flat, w.gate_proj), config.mlp_activation)
    up = gemm(x_flat, w.up_proj)
    fused = (gate.astype(np.float32) * up.astype(np.float32)).astype(x.dtype)
    return gemm(fused, w.down_proj).reshape(B, S, H)


def llama_decoder_layer(x, w, config, *, attn_impl=sdpa):
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    x = x + _attention_block(h, w, config, attn_impl)
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _mlp_block(h, w, config)
    return x


def llama_forward(input_ids, weights, config, *, attn_impl=sdpa):
    """Dense (non-paged) forward. Returns logits of shape ``(B, S, vocab_size)``."""
    x = embedding_lookup(weights.embed_tokens, input_ids)
    if config.embed_scale != 1.0:
        x = (x.astype(np.float32) * config.embed_scale).astype(x.dtype)
    for layer in weights.layers:
        x = llama_decoder_layer(x, layer, config, attn_impl=attn_impl)
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    return gemm(x.reshape(B * S, H), weights.lm_head).reshape(B, S, config.vocab_size)


# --- paged prefill / decode (M4) ---------------------------------------------

def _project_qkv_paged(x_flat, w, config, positions, B, S):
    """Project x → Q/K/V, reshape, apply RoPE.

    Returns:
      q_paged: (B, S, num_q_heads * head_dim) — flat-head form for paged_attention
      k_kv:    (B, S, num_kv_heads, head_dim) — for paged_kv_append
      v_kv:    (B, S, num_kv_heads, head_dim)
    """
    n_q, n_kv, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = gemm(x_flat, w.q_proj).reshape(B, S, n_q, D).transpose(0, 2, 1, 3)  # (B, n_q, S, D)
    k = gemm(x_flat, w.k_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)  # (B, n_kv, S, D)
    v = gemm(x_flat, w.v_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)  # (B, n_kv, S, D)
    # RoPE expects (..., S, D); the per-position frequencies need the right
    # absolute positions, which differ per request during continuous batching.
    # When all requests share the same starting position (uniform prefill or
    # uniform decode step), we can apply rope batched.
    q = rope(q, positions=positions, theta=config.rope_theta)
    k = rope(k, positions=positions, theta=config.rope_theta)
    # Shape K/V back to (B, S, n_kv, D) for paged_kv_append.
    k_kv = k.transpose(0, 2, 1, 3)
    v_kv = v.transpose(0, 2, 1, 3)
    # Q goes into paged_attention as (B, S, n_q*D).
    q_flat = q.transpose(0, 2, 1, 3).reshape(B, S, n_q * D)
    return q_flat, k_kv, v_kv


def _attention_block_paged(x, w, config, *, layer_idx, state, query_positions, query_lens_after):
    """Attention block over paged KV.

    ``query_positions``: (S,) absolute positions of the new query tokens (used
        for RoPE). All requests in the batch share the same positions in this
        call (uniform prefill or uniform decode).
    ``query_lens_after``: (B,) sequence length AFTER appending these queries.
    """
    B, S, H = x.shape
    n_q, n_kv, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    dt = x.dtype
    x_flat = x.reshape(B * S, H)
    q_flat, k_kv, v_kv = _project_qkv_paged(x_flat, w, config, query_positions, B, S)

    # seq_lens BEFORE this append are state.seq_lens; AFTER they will be
    # query_lens_after. We append using the BEFORE values.
    seq_lens_before = (query_lens_after - S).astype(np.int32)
    paged_kv_append(
        state.cache_k[layer_idx], state.cache_v[layer_idx],
        k_kv.astype(state.cache_k[layer_idx].dtype),
        v_kv.astype(state.cache_v[layer_idx].dtype),
        block_table=state.block_table,
        seq_lens=seq_lens_before,
        block_size=state.block_size,
    )

    out = _paged_attention(
        q_flat, state.cache_k[layer_idx], state.cache_v[layer_idx],
        block_table=state.block_table,
        seq_lens=query_lens_after,
        block_size=state.block_size,
        num_q_heads=n_q, num_kv_heads=n_kv, head_dim=D,
        causal=True,
    )
    out = out.astype(dt).reshape(B * S, n_q * D)
    return gemm(out, w.o_proj).reshape(B, S, H)


def _decoder_layer_paged(x, w, config, *, layer_idx, state, query_positions, query_lens_after):
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    x = x + _attention_block_paged(
        h, w, config,
        layer_idx=layer_idx, state=state,
        query_positions=query_positions, query_lens_after=query_lens_after,
    )
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _mlp_block(h, w, config)
    return x


def llama_prefill(
    input_ids: np.ndarray,
    weights: LlamaWeights,
    config: LlamaConfig,
    *,
    state: PagedKVState,
    position_offset: int = 0,
) -> np.ndarray:
    """Prefill: process a uniform-length chunk for every request and write KV.

    ``input_ids``: ``(B, S)`` — tokens to prefill, same length per request.
    Caller must reserve enough physical blocks per request to cover
    ``position_offset + S``.

    ``position_offset``: starting absolute position for these tokens. Defaults
    to 0 (the M2/M4 path: prompt prefill from scratch). With a non-zero
    offset the call is a *suffix prefill* — the M5 prefix-cache path:
    cached blocks already populate ``state.cache_k/v`` for positions
    ``[0, position_offset)``, and we prefill the suffix
    ``[position_offset, position_offset + S)`` against that history.

    All requests in this call must share the same ``position_offset`` —
    the scheduler groups by ``(position_offset, S)`` to batch uniformly.
    Returns logits ``(B, S, vocab_size)``.
    """
    B, S = input_ids.shape
    if not np.all(state.seq_lens == position_offset):
        raise ValueError(
            f"llama_prefill: state.seq_lens must equal position_offset "
            f"({position_offset}); got {state.seq_lens.tolist()}"
        )
    query_positions = np.arange(
        position_offset, position_offset + S, dtype=np.int64,
    )
    query_lens_after = np.full((B,), position_offset + S, dtype=np.int32)

    x = embedding_lookup(weights.embed_tokens, input_ids)
    if config.embed_scale != 1.0:
        x = (x.astype(np.float32) * config.embed_scale).astype(x.dtype)
    for i, layer in enumerate(weights.layers):
        x = _decoder_layer_paged(
            x, layer, config,
            layer_idx=i, state=state,
            query_positions=query_positions, query_lens_after=query_lens_after,
        )
    state.seq_lens[:] = query_lens_after
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B_, S_, H = x.shape
    return gemm(x.reshape(B_ * S_, H), weights.lm_head).reshape(B_, S_, config.vocab_size)


def llama_decode_step(
    token_ids: np.ndarray,
    weights: LlamaWeights,
    config: LlamaConfig,
    *,
    state: PagedKVState,
) -> np.ndarray:
    """Decode: process one new token per request, append KV, return logits.

    ``token_ids``: ``(B,)`` — one new token per request.
    ``state.seq_lens`` is the number of tokens already in the cache; this
    call appends one and returns next-token logits ``(B, vocab_size)``.
    """
    B = token_ids.shape[0]
    if state.seq_lens.shape[0] != B:
        raise ValueError("token_ids batch must match state.seq_lens batch")
    # All requests in a single decode step share the position-axis of length 1
    # but differ in absolute position. paged_attention handles per-request
    # offsets via seq_lens; for RoPE we can still apply with a length-1 axis
    # and a per-request position vector — but our `rope` takes one positions
    # array applied across the batch axis. Workaround: in this reference
    # impl we apply per-request positions by invoking RoPE on per-request
    # slices of the projected Q/K. That is wasteful but readable; the lhsil
    # kernel does it in one launch.
    x = embedding_lookup(weights.embed_tokens, token_ids).reshape(B, 1, config.hidden_size)
    if config.embed_scale != 1.0:
        x = (x.astype(np.float32) * config.embed_scale).astype(x.dtype)
    query_lens_after = state.seq_lens + 1

    for i, layer in enumerate(weights.layers):
        x = _decoder_layer_paged_per_request_rope(
            x, layer, config,
            layer_idx=i, state=state,
            query_lens_after=query_lens_after,
        )

    state.seq_lens[:] = query_lens_after
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    return gemm(
        x.reshape(B, config.hidden_size), weights.lm_head
    ).reshape(B, config.vocab_size)


def _decoder_layer_paged_per_request_rope(x, w, config, *, layer_idx, state, query_lens_after):
    """Decode-step decoder layer with per-request RoPE positions.

    Each request's single new query token sits at a different absolute
    position (``state.seq_lens[b]``), so RoPE must be applied per request.
    """
    B, S, H = x.shape  # S == 1 in decode
    n_q, n_kv, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    dt = x.dtype

    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    x_flat = h.reshape(B * S, H)
    q_h = gemm(x_flat, w.q_proj).reshape(B, S, n_q, D).transpose(0, 2, 1, 3)  # (B, n_q, S, D)
    k_h = gemm(x_flat, w.k_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)
    v_h = gemm(x_flat, w.v_proj).reshape(B, S, n_kv, D).transpose(0, 2, 1, 3)

    # Per-request RoPE — each request's query is at its own absolute position.
    seq_lens_before = (query_lens_after - 1).astype(np.int64)
    for b in range(B):
        pos = np.array([seq_lens_before[b]], dtype=np.int64)
        q_h[b : b + 1] = rope(q_h[b : b + 1], positions=pos, theta=config.rope_theta)
        k_h[b : b + 1] = rope(k_h[b : b + 1], positions=pos, theta=config.rope_theta)

    # Append the new K/V via paged_kv_append.
    k_kv = k_h.transpose(0, 2, 1, 3)
    v_kv = v_h.transpose(0, 2, 1, 3)
    paged_kv_append(
        state.cache_k[layer_idx], state.cache_v[layer_idx],
        k_kv.astype(state.cache_k[layer_idx].dtype),
        v_kv.astype(state.cache_v[layer_idx].dtype),
        block_table=state.block_table,
        seq_lens=seq_lens_before.astype(np.int32),
        block_size=state.block_size,
    )

    q_flat = q_h.transpose(0, 2, 1, 3).reshape(B, S, n_q * D)
    out = _paged_attention(
        q_flat, state.cache_k[layer_idx], state.cache_v[layer_idx],
        block_table=state.block_table,
        seq_lens=query_lens_after,
        block_size=state.block_size,
        num_q_heads=n_q, num_kv_heads=n_kv, head_dim=D,
        causal=True,
    )
    out = out.astype(dt).reshape(B * S, n_q * D)
    attn_out = gemm(out, w.o_proj).reshape(B, S, H)
    x = x + attn_out

    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    return x + _mlp_block(h, w, config)


__all__ = [
    "LlamaConfig",
    "LlamaLayerWeights",
    "LlamaWeights",
    "PagedKVState",
    "init_random_weights",
    "alloc_paged_kv_state",
    "llama_decoder_layer",
    "llama_forward",
    "llama_prefill",
    "llama_decode_step",
]
