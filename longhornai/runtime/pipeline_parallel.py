"""Pipeline parallelism (PLAN.md §3 Phase 6 / §8 M7).

Pipeline-parallel (PP) inference partitions the model's *layers* across
ranks: rank 0 owns layers ``[0, L/N)``, rank 1 owns ``[L/N, 2L/N)``, etc.
Each rank receives the previous rank's hidden state, runs its layer-stage,
and forwards to the next rank. The first rank owns the embedding; the
last rank owns the final norm + LM head.

This module simulates N virtual ranks sequentially in one process; the
contract is the *output* — :func:`llama_forward_pp` produces the same
logits as :func:`longhornai.models.llama_forward`. lhsil's implementation
runs each stage on a different device with point-to-point sends between
them; for batched inference the canonical optimization is **micro-batch
pipelining** (1F1B), which doesn't change the output.

PP completes the distributed-inference triad (TP + EP + PP) per
PLAN.md §3 Phase 6 / §8 M7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..kernels import embedding_lookup, gemm, rmsnorm
from ..models.llama import (
    AttnImpl,
    LlamaConfig,
    LlamaWeights,
    LlamaLayerWeights,
    llama_decoder_layer,
    sdpa,
)


@dataclass
class PPStage:
    """One pipeline stage's owned tensors.

    Stage 0 holds the embedding; stage N-1 holds the final norm + LM head.
    Every stage holds its slice of the decoder layers.
    """

    rank: int
    layers: List[LlamaLayerWeights]
    embed_tokens: np.ndarray | None = None    # only on rank 0
    final_norm: np.ndarray | None = None      # only on rank N-1
    lm_head: np.ndarray | None = None         # only on rank N-1


@dataclass
class PPWeights:
    """Layer-partitioned weights across pipeline ranks."""

    stages: List[PPStage]
    num_ranks: int
    embed_scale: float = 1.0


def shard_llama_for_pp(
    weights: LlamaWeights, *, num_ranks: int, embed_scale: float = 1.0,
) -> PPWeights:
    """Partition decoder layers across ``num_ranks`` pipeline stages.

    ``num_hidden_layers`` must be divisible by ``num_ranks``. Stage 0 also
    owns the embedding; the last stage owns the final norm + LM head.
    """
    L = len(weights.layers)
    if L % num_ranks != 0:
        raise ValueError(
            f"shard_llama_for_pp: num_hidden_layers ({L}) must be divisible "
            f"by num_ranks ({num_ranks})"
        )
    per_stage = L // num_ranks
    stages: List[PPStage] = []
    for r in range(num_ranks):
        layer_slice = weights.layers[r * per_stage : (r + 1) * per_stage]
        stages.append(PPStage(
            rank=r,
            layers=layer_slice,
            embed_tokens=weights.embed_tokens if r == 0 else None,
            final_norm=weights.final_norm if r == num_ranks - 1 else None,
            lm_head=weights.lm_head if r == num_ranks - 1 else None,
        ))
    return PPWeights(stages=stages, num_ranks=num_ranks, embed_scale=embed_scale)


def llama_forward_pp(
    input_ids: np.ndarray,
    weights: PPWeights,
    config: LlamaConfig,
    *,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Pipeline-parallel Llama forward — produces the same logits as
    :func:`longhornai.models.llama_forward` on a single device.
    """
    # Stage 0: embed.
    stage0 = weights.stages[0]
    if stage0.embed_tokens is None:
        raise ValueError("Stage 0 must own embed_tokens")
    x = embedding_lookup(stage0.embed_tokens, input_ids)
    if config.embed_scale != 1.0:
        x = (x.astype(np.float32) * config.embed_scale).astype(x.dtype)

    # Run each stage's layers in turn. The "send" between ranks is a no-op
    # in this simulator — `x` is the wire-level activation that flows.
    for stage in weights.stages:
        for layer in stage.layers:
            x = llama_decoder_layer(x, layer, config, attn_impl=attn_impl)

    # Final stage: norm + LM head.
    last = weights.stages[-1]
    if last.final_norm is None or last.lm_head is None:
        raise ValueError("Last stage must own final_norm + lm_head")
    x = rmsnorm(x, weight=last.final_norm, eps=config.rms_norm_eps)
    B, S, H = x.shape
    return gemm(x.reshape(B * S, H), last.lm_head).reshape(
        B, S, config.vocab_size,
    )


def micro_batch_forward_pp(
    input_ids: np.ndarray,
    weights: PPWeights,
    config: LlamaConfig,
    *,
    num_micro_batches: int,
    attn_impl: AttnImpl = sdpa,
) -> np.ndarray:
    """Same forward, computed via micro-batch pipelining (1F1B).

    The batch is split into ``num_micro_batches`` chunks; each chunk flows
    down the pipeline in turn. Numerically identical to
    :func:`llama_forward_pp` — micro-batching is the bubble-reduction
    optimization that lhsil's runtime applies, not a math change.

    ``input_ids.shape[0]`` (batch size) must be divisible by
    ``num_micro_batches``.
    """
    B = input_ids.shape[0]
    if B % num_micro_batches != 0:
        raise ValueError(
            f"micro_batch_forward_pp: batch size ({B}) must be divisible by "
            f"num_micro_batches ({num_micro_batches})"
        )
    chunks = np.split(input_ids, num_micro_batches, axis=0)
    outs = [
        llama_forward_pp(chunk, weights, config, attn_impl=attn_impl)
        for chunk in chunks
    ]
    return np.concatenate(outs, axis=0)


__all__ = [
    "PPStage",
    "PPWeights",
    "shard_llama_for_pp",
    "llama_forward_pp",
    "micro_batch_forward_pp",
]
