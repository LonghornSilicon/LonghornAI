"""CPU implementations for the MoE family."""

from __future__ import annotations

import numpy as np

from ...backends.cpu import cpu
from .reference import (
    ref_moe_combine,
    ref_moe_dispatch,
    ref_moe_router,
    ref_moe_top_k,
)


@cpu.register("moe_router")
def moe_router(x, gate_weight):
    return ref_moe_router(x, gate_weight)


@cpu.register("moe_top_k")
def moe_top_k(logits, *, k, normalize=True):
    return ref_moe_top_k(logits, k=k, normalize=normalize)


@cpu.register("moe_dispatch")
def moe_dispatch(x, expert_ids, *, num_experts):
    return ref_moe_dispatch(x, expert_ids, num_experts=num_experts)


@cpu.register("moe_combine")
def moe_combine(expert_outputs, weights, recovery, *, n_tokens, hidden):
    return ref_moe_combine(
        expert_outputs, weights, recovery, n_tokens=n_tokens, hidden=hidden,
    )
