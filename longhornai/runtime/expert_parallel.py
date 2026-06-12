"""Expert parallelism (PLAN.md §3 Phase 5 / §8 M6).

Expert-parallel (EP) inference shards the **experts** of a Sparse-MoE
layer across N ranks: each rank owns ``num_local_experts / N`` experts.
Tokens routed to a rank's local experts run there; tokens routed to
remote experts must be exchanged via **all-to-all**.

The forward looks like:

1. Every rank runs the router over the full batch (router weights are
   replicated; this is cheap relative to the experts).
2. Each rank computes top-K and produces ``(token, k) → expert_id``.
3. Tokens are bucketed by *target rank* (which rank owns the chosen
   expert). An ``all_to_all`` ships each rank's outbound bucket to its
   destination, so each rank now holds the tokens its local experts
   need.
4. Local SwiGLU per expert.
5. A second ``all_to_all`` ships expert outputs back to each token's
   originating rank.
6. ``moe_combine`` un-permutes and weighted-sums per-token.

PLAN.md §8 M6 exit gate: expert-parallel functional, with EP-Mixtral
matching the single-device :func:`mixtral_forward` bit-tight under
tolerance.

This module simulates N virtual ranks sequentially; the contract — the
output logits — must equal the single-device forward's output. lhsil's
implementation replaces the in-process all-to-all with the real
interconnect call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..kernels import (
    all_to_all,
    embedding_lookup,
    gemm,
    moe_combine,
    moe_dispatch,
    moe_router,
    moe_top_k,
    rmsnorm,
    sdpa,
    silu,
)
from ..models.llama import AttnImpl, _attention_block, LlamaLayerWeights
from ..models.mixtral import (
    MixtralConfig,
    MixtralExpertWeights,
    MixtralWeights,
)


# --- per-layer EP weights ----------------------------------------------------

@dataclass
class EPLayerWeights:
    """Per-layer weights, sharded for expert parallelism.

    Attention weights are *replicated* — TP could shard them too but EP
    here only splits experts. Expert weights are sharded: rank r owns
    ``experts_per_rank`` consecutive experts.
    """

    input_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    post_attn_norm: np.ndarray
    router_gate: np.ndarray
    # experts_per_rank lists — each entry holds rank r's owned experts.
    experts: List[List[MixtralExpertWeights]]


@dataclass
class EPWeights:
    embed_tokens: np.ndarray
    layers: List[EPLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    num_ranks: int
    experts_per_rank: int


def shard_mixtral_for_ep(
    weights: MixtralWeights, *, num_ranks: int,
) -> EPWeights:
    """Shard Mixtral experts across ``num_ranks`` ranks.

    ``num_local_experts`` must be divisible by ``num_ranks``. Each rank
    owns ``num_local_experts / num_ranks`` consecutive experts (rank 0:
    experts [0, n), rank 1: [n, 2n), …).
    """
    if not weights.layers:
        raise ValueError("Mixtral has no layers")
    E = len(weights.layers[0].experts)
    if E % num_ranks != 0:
        raise ValueError(
            f"shard_mixtral_for_ep: num_local_experts ({E}) must be "
            f"divisible by num_ranks ({num_ranks})"
        )
    per_rank = E // num_ranks

    layers: List[EPLayerWeights] = []
    for layer in weights.layers:
        sharded = [
            layer.experts[r * per_rank : (r + 1) * per_rank]
            for r in range(num_ranks)
        ]
        layers.append(EPLayerWeights(
            input_norm=layer.input_norm,
            q_proj=layer.q_proj, k_proj=layer.k_proj, v_proj=layer.v_proj,
            o_proj=layer.o_proj,
            post_attn_norm=layer.post_attn_norm,
            router_gate=layer.router_gate,
            experts=sharded,
        ))
    return EPWeights(
        embed_tokens=weights.embed_tokens,
        layers=layers,
        final_norm=weights.final_norm,
        lm_head=weights.lm_head,
        num_ranks=num_ranks,
        experts_per_rank=per_rank,
    )


# --- EP MoE FFN --------------------------------------------------------------

def _expert_swiglu(x_for_expert: np.ndarray, expert: MixtralExpertWeights):
    if x_for_expert.shape[0] == 0:
        H = expert.down_proj.shape[1]
        return np.zeros((0, H), dtype=x_for_expert.dtype)
    gate = silu(gemm(x_for_expert, expert.gate_proj))
    up = gemm(x_for_expert, expert.up_proj)
    fused = (gate.astype(np.float32) * up.astype(np.float32)).astype(x_for_expert.dtype)
    return gemm(fused, expert.down_proj)


def _ep_moe_ffn(
    x: np.ndarray, w: EPLayerWeights, config: MixtralConfig,
    *, num_ranks: int, experts_per_rank: int,
) -> np.ndarray:
    """Sparse-MoE FFN with experts sharded across ranks.

    The reference simulates the multi-rank world by:
    1. Running the router over the full batch (router replicated).
    2. Building an expert-major permutation as in single-device.
    3. Splitting the permuted tokens into per-rank buckets — rank r gets
       the slab destined for experts [r*per_rank, (r+1)*per_rank).
    4. Running local SwiGLU per expert on each rank.
    5. Reassembling expert outputs back into the original permutation order.
    6. Combining via :func:`moe_combine`.

    Steps 3 + 5 are the all-to-all token routing; in this single-process
    simulation the "exchange" is a list slice + concatenate. lhsil performs
    a real interconnect all-to-all here.
    """
    B, S, H = x.shape
    tokens = B * S
    x_flat = x.reshape(tokens, H)
    E = num_ranks * experts_per_rank

    logits = moe_router(x_flat, w.router_gate)
    expert_ids, weights = moe_top_k(
        logits, k=config.num_experts_per_tok, normalize=True,
    )
    permuted, offsets, recovery = moe_dispatch(
        x_flat, expert_ids, num_experts=E,
    )

    # Per-rank slab boundaries in the (tokens*k) permutation: rank r covers
    # experts [r*per_rank, (r+1)*per_rank), so its row range is
    # [offsets[r*per_rank], offsets[(r+1)*per_rank]).
    rank_starts = [int(offsets[r * experts_per_rank]) for r in range(num_ranks)]
    rank_ends = [int(offsets[(r + 1) * experts_per_rank]) for r in range(num_ranks)]

    # Step 3: simulate the dispatch all-to-all by slicing.
    rank_inputs = [permuted[rank_starts[r]:rank_ends[r]] for r in range(num_ranks)]

    # Step 4: local SwiGLU per expert on each rank.
    rank_outputs: List[np.ndarray] = []
    for r in range(num_ranks):
        local_in = rank_inputs[r]
        local_out = np.empty_like(local_in)
        # Within the rank's slab, expert e (local index) covers a sub-slab.
        local_offsets = [
            int(offsets[r * experts_per_rank + e]) - rank_starts[r]
            for e in range(experts_per_rank + 1)
        ]
        for e_local in range(experts_per_rank):
            s, t = local_offsets[e_local], local_offsets[e_local + 1]
            if t > s:
                local_out[s:t] = _expert_swiglu(local_in[s:t], w.experts[r][e_local])
        rank_outputs.append(local_out)

    # Step 5: reassemble — concatenating in rank order recovers the full
    # expert-major permutation. (This is the *return* all-to-all; in
    # production it ships outputs back to each token's home rank.)
    expert_outs = np.concatenate(rank_outputs, axis=0) if rank_outputs else permuted

    # Step 6: combine.
    combined = moe_combine(
        expert_outs, weights, recovery, n_tokens=tokens, hidden=H,
    )
    return combined.reshape(B, S, H)


def _ep_decoder_layer(x, w, config, attn_impl, num_ranks, experts_per_rank):
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    attn_view = LlamaLayerWeights(
        input_norm=w.input_norm,
        q_proj=w.q_proj, k_proj=w.k_proj, v_proj=w.v_proj, o_proj=w.o_proj,
        post_attn_norm=w.post_attn_norm,
        gate_proj=np.zeros((1, 1), dtype=x.dtype),
        up_proj=np.zeros((1, 1), dtype=x.dtype),
        down_proj=np.zeros((1, 1), dtype=x.dtype),
    )
    x = x + _attention_block(h, attn_view, config.to_llama(), attn_impl)
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _ep_moe_ffn(h, w, config,
                         num_ranks=num_ranks, experts_per_rank=experts_per_rank)
    return x


def mixtral_forward_ep(
    input_ids: np.ndarray,
    weights: EPWeights,
    config: MixtralConfig,
    *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Expert-parallel Mixtral forward — same output as single-device
    :func:`mixtral_forward`, with experts sharded across ranks."""
    x = embedding_lookup(weights.embed_tokens, input_ids)
    for layer in weights.layers:
        x = _ep_decoder_layer(
            x, layer, config, attn_impl,
            weights.num_ranks, weights.experts_per_rank,
        )
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    return gemm(x.reshape(B * S, H), weights.lm_head).reshape(B, S, config.vocab_size)


__all__ = [
    "EPLayerWeights",
    "EPWeights",
    "shard_mixtral_for_ep",
    "mixtral_forward_ep",
]
