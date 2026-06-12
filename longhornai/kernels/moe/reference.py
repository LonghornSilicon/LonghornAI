"""Float64 references for the MoE family.

The references encode the routing semantics that every backend impl must
match — softmax-over-top-k weight normalization, expert-major token
permutation order, and the recovery indexing used for un-permute.
"""

from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def ref_moe_router(x, gate_weight):
    return (_f64(x) @ _f64(gate_weight)).astype(x.dtype)


def ref_moe_top_k(logits, *, k, normalize=True):
    """Top-k experts per token + softmax-normalized weights."""
    L = _f64(logits)
    # argpartition is faster than full sort but we want sorted indices for
    # determinism; use argsort and slice.
    sorted_idx = np.argsort(-L, axis=-1)                # (tokens, num_experts)
    top_idx = sorted_idx[:, :k].astype(np.int32)
    top_logits = np.take_along_axis(L, top_idx, axis=-1)
    if normalize:
        m = np.max(top_logits, axis=-1, keepdims=True)
        e = np.exp(top_logits - m)
        weights = (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)
    else:
        weights = top_logits.astype(np.float32)
    return top_idx, weights


def ref_moe_dispatch(x, expert_ids, *, num_experts):
    """Expert-major permutation: scatter ``x`` rows by ``expert_ids``.

    Returns ``(permuted_x, expert_offsets, recovery)`` where ``recovery`` is
    the flat index into ``(tokens * k,)`` that maps each output row back to
    its slot in the input ``expert_ids``.
    """
    tokens, k = expert_ids.shape
    flat_ids = expert_ids.reshape(-1)                    # (tokens*k,)
    flat_token = np.repeat(np.arange(tokens), k)         # (tokens*k,)
    flat_slot = np.tile(np.arange(k), tokens)            # (tokens*k,)
    flat_index = flat_token * k + flat_slot              # (tokens*k,)
    # Stable sort by expert id so per-expert order is reproducible.
    perm = np.argsort(flat_ids, kind="stable")
    permuted_token = flat_token[perm]
    permuted_x = _f64(x)[permuted_token].astype(x.dtype)
    recovery = flat_index[perm].astype(np.int32)         # (tokens*k,)
    # Per-expert offsets via cumulative count.
    counts = np.bincount(flat_ids.astype(np.int64), minlength=num_experts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return permuted_x, offsets, recovery


def ref_moe_combine(expert_outputs, weights, recovery, *, n_tokens, hidden):
    """Un-permute ``expert_outputs`` and weighted-sum back to per-token form."""
    out_dt = expert_outputs.dtype
    eo = _f64(expert_outputs)                            # (tokens*k, hidden)
    w_flat = _f64(weights).reshape(-1)                   # (tokens*k,)
    out = np.zeros((n_tokens, hidden), dtype=np.float64)
    for row, dest_flat in enumerate(recovery):
        token = int(dest_flat) // weights.shape[-1]
        slot = int(dest_flat) % weights.shape[-1]
        out[token] += eo[row] * w_flat[token * weights.shape[-1] + slot]
    return out.astype(out_dt)


REFERENCES = {
    "moe_router": ref_moe_router,
    "moe_top_k": ref_moe_top_k,
    "moe_dispatch": ref_moe_dispatch,
    "moe_combine": ref_moe_combine,
}

__all__ = [
    "REFERENCES",
    "ref_moe_router",
    "ref_moe_top_k",
    "ref_moe_dispatch",
    "ref_moe_combine",
]
