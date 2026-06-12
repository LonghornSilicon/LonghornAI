"""Tensor parallelism (PLAN.md §3 Phase 6 / §8 M6).

Tensor-parallel (TP) inference shards each linear projection across N
ranks: input or output dimensions are split, so each rank holds 1/N of
the weights and contributes 1/N of the compute. The two patterns:

* **Column-parallel** (split output dim): each rank holds W[:, r*S:(r+1)*S].
  Each rank computes its slice locally; the outputs concatenate to form
  the full result. No inter-rank communication on the forward.
* **Row-parallel** (split input dim): each rank holds W[r*S:(r+1)*S, :].
  The inputs are split too; each rank's local matmul is a partial sum;
  an **all-reduce** at the end combines them.

The Llama attention block uses column-parallel for QKV projections (output
sharded along the head axis) and row-parallel for the output projection.
The MLP uses column-parallel for gate/up and row-parallel for down.
PLAN.md §8 M6 exit gate component: TP functional, with TP-Llama matching
the single-device forward bit-tight under tolerance.

This module simulates N virtual ranks on a single CPU process; lhsil's
backend lowering swaps the in-process collective for the real
interconnect call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..kernels import (
    all_reduce,
    embedding_lookup,
    gemm,
    rmsnorm,
    rope,
    sdpa,
    silu,
)
from ..models.llama import (
    AttnImpl,
    LlamaConfig,
    LlamaWeights,
    _gated_activation,
)


# --- weight sharding ---------------------------------------------------------

def shard_column(weight: np.ndarray, num_ranks: int) -> List[np.ndarray]:
    """Split a ``(in, out)`` weight along the output dim across ranks."""
    if weight.shape[1] % num_ranks != 0:
        raise ValueError(
            f"shard_column: out-dim {weight.shape[1]} not divisible by "
            f"{num_ranks} ranks"
        )
    return [w.copy() for w in np.split(weight, num_ranks, axis=1)]


def shard_row(weight: np.ndarray, num_ranks: int) -> List[np.ndarray]:
    """Split a ``(in, out)`` weight along the input dim across ranks."""
    if weight.shape[0] % num_ranks != 0:
        raise ValueError(
            f"shard_row: in-dim {weight.shape[0]} not divisible by "
            f"{num_ranks} ranks"
        )
    return [w.copy() for w in np.split(weight, num_ranks, axis=0)]


# --- per-layer TP weights ----------------------------------------------------

@dataclass
class TPLayerWeights:
    """Per-layer weights, sharded for tensor parallelism.

    Each list has ``num_ranks`` entries — rank r's shard of the named tensor.
    QKV / gate / up are column-parallel; o_proj / down_proj are row-parallel.
    """

    input_norm: np.ndarray            # replicated
    q_proj: List[np.ndarray]          # column-parallel  (H, n_q*D / R)
    k_proj: List[np.ndarray]          # column-parallel  (H, n_kv*D / R)
    v_proj: List[np.ndarray]          # column-parallel  (H, n_kv*D / R)
    o_proj: List[np.ndarray]          # row-parallel     (n_q*D / R, H)
    post_attn_norm: np.ndarray        # replicated
    gate_proj: List[np.ndarray]       # column-parallel  (H, I / R)
    up_proj: List[np.ndarray]         # column-parallel  (H, I / R)
    down_proj: List[np.ndarray]       # row-parallel     (I / R, H)


@dataclass
class TPWeights:
    """Full-model TP weights — embed + per-layer + lm_head."""

    embed_tokens: np.ndarray          # replicated (small)
    layers: List[TPLayerWeights]
    final_norm: np.ndarray            # replicated
    lm_head: List[np.ndarray]         # column-parallel  (H, vocab / R)
    num_ranks: int


def shard_llama_for_tp(
    weights: LlamaWeights, *, num_ranks: int,
) -> TPWeights:
    """Re-shard a single-device :class:`LlamaWeights` for TP execution.

    Sharding constraints:
    * QKV column-parallel along the head dim — both ``num_attention_heads``
      and ``num_kv_heads`` must be divisible by ``num_ranks``.
    * MLP column-parallel along ``intermediate_size`` — must be divisible.
    * LM head column-parallel along vocab — must be divisible.
    """
    layers: List[TPLayerWeights] = []
    for layer in weights.layers:
        layers.append(TPLayerWeights(
            input_norm=layer.input_norm,
            q_proj=shard_column(layer.q_proj, num_ranks),
            k_proj=shard_column(layer.k_proj, num_ranks),
            v_proj=shard_column(layer.v_proj, num_ranks),
            o_proj=shard_row(layer.o_proj, num_ranks),
            post_attn_norm=layer.post_attn_norm,
            gate_proj=shard_column(layer.gate_proj, num_ranks),
            up_proj=shard_column(layer.up_proj, num_ranks),
            down_proj=shard_row(layer.down_proj, num_ranks),
        ))
    return TPWeights(
        embed_tokens=weights.embed_tokens,
        layers=layers,
        final_norm=weights.final_norm,
        lm_head=shard_column(weights.lm_head, num_ranks),
        num_ranks=num_ranks,
    )


# --- TP forward --------------------------------------------------------------

def _tp_attention(x, w: TPLayerWeights, config: LlamaConfig, attn_impl, R: int):
    """Tensor-parallel attention block.

    Heads are partitioned across ranks via the column-parallel QKV shards.
    Each rank computes attention over its slice of heads, then projects via
    its row-parallel ``o_proj`` shard — and an all-reduce sums the partial
    output projections back to the full hidden state.
    """
    B, S, H = x.shape
    n_q, n_kv, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    if n_q % R != 0 or n_kv % R != 0:
        raise ValueError(
            f"TP needs num_attention_heads ({n_q}) and num_kv_heads ({n_kv}) "
            f"divisible by num_ranks ({R})"
        )
    n_q_per = n_q // R
    n_kv_per = n_kv // R

    # Each rank: project x → its slice of Q/K/V; do RoPE; broadcast K/V for
    # GQA; SDPA; project via its row-parallel o_proj slice. The o_proj
    # outputs partial-sum across ranks and we all-reduce.
    partial_outputs = []
    x_flat = x.reshape(B * S, H)
    for r in range(R):
        q_r = gemm(x_flat, w.q_proj[r]).reshape(B, S, n_q_per, D).transpose(0, 2, 1, 3)
        k_r = gemm(x_flat, w.k_proj[r]).reshape(B, S, n_kv_per, D).transpose(0, 2, 1, 3)
        v_r = gemm(x_flat, w.v_proj[r]).reshape(B, S, n_kv_per, D).transpose(0, 2, 1, 3)
        q_r = rope(q_r, theta=config.rope_theta)
        k_r = rope(k_r, theta=config.rope_theta)
        if n_kv_per != n_q_per:
            repeats = n_q_per // n_kv_per
            k_r = np.repeat(k_r, repeats, axis=1)
            v_r = np.repeat(v_r, repeats, axis=1)
        attn_r = attn_impl(q_r, k_r, v_r, causal=True)              # (B, n_q_per, S, D)
        flat_r = attn_r.transpose(0, 2, 1, 3).reshape(B * S, n_q_per * D).astype(x.dtype)
        partial_outputs.append(gemm(flat_r, w.o_proj[r]))            # (B*S, H)

    summed_shards = all_reduce(partial_outputs, op="sum")
    return summed_shards[0].reshape(B, S, H)


def _tp_mlp(x, w: TPLayerWeights, config: LlamaConfig, R: int):
    """Tensor-parallel SwiGLU MLP.

    Gate / up are column-parallel; down_proj is row-parallel; the down
    output partial-sums across ranks via all_reduce.
    """
    B, S, H = x.shape
    x_flat = x.reshape(B * S, H)
    partials = []
    for r in range(R):
        gate_r = _gated_activation(gemm(x_flat, w.gate_proj[r]), config.mlp_activation)
        up_r = gemm(x_flat, w.up_proj[r])
        fused_r = (gate_r.astype(np.float32) * up_r.astype(np.float32)).astype(x.dtype)
        partials.append(gemm(fused_r, w.down_proj[r]))
    summed = all_reduce(partials, op="sum")
    return summed[0].reshape(B, S, H)


def _tp_decoder_layer(x, w: TPLayerWeights, config: LlamaConfig, attn_impl, R: int):
    h = rmsnorm(x, weight=w.input_norm, eps=config.rms_norm_eps)
    x = x + _tp_attention(h, w, config, attn_impl, R)
    h = rmsnorm(x, weight=w.post_attn_norm, eps=config.rms_norm_eps)
    x = x + _tp_mlp(h, w, config, R)
    return x


def llama_forward_tp(
    input_ids: np.ndarray,
    weights: TPWeights,
    config: LlamaConfig,
    *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Tensor-parallel Llama forward — produces the same logits as
    :func:`longhornai.models.llama_forward` on a single device.
    """
    R = weights.num_ranks
    x = embedding_lookup(weights.embed_tokens, input_ids)
    if config.embed_scale != 1.0:
        x = (x.astype(np.float32) * config.embed_scale).astype(x.dtype)
    for layer in weights.layers:
        x = _tp_decoder_layer(x, layer, config, attn_impl, R)
    x = rmsnorm(x, weight=weights.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    # LM head is column-parallel over vocab; each rank produces a vocab slice
    # and we concatenate to form the full logit tensor.
    x_flat = x.reshape(B * S, H)
    head_shards = [gemm(x_flat, weights.lm_head[r]) for r in range(R)]
    logits = np.concatenate(head_shards, axis=-1)
    return logits.reshape(B, S, config.vocab_size)


__all__ = [
    "TPLayerWeights",
    "TPWeights",
    "shard_column",
    "shard_row",
    "shard_llama_for_tp",
    "llama_forward_tp",
]
