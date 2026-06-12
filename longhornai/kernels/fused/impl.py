"""CPU implementations for the fused-kernel family.

The CPU reference computes each fused op as the equivalent un-fused chain
in float64; lhsil's implementation issues the math as a single launch
that streams operands through registers between sub-ops, cutting HBM
round-trips. Either way the *output* must match the reference.
"""

from __future__ import annotations

import numpy as np

from ...backends.cpu import cpu
from .reference import (
    ref_attention_output_proj,
    ref_gated_mlp,
    ref_rmsnorm_qkv,
)


@cpu.register("rmsnorm_qkv")
def rmsnorm_qkv(x, *, norm_weight, q_weight, k_weight, v_weight, eps=1e-5):
    return ref_rmsnorm_qkv(
        x, norm_weight=norm_weight,
        q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, eps=eps,
    )


@cpu.register("attention_output_proj")
def attention_output_proj(q, k, v, *, o_weight, causal=False, scale=None):
    return ref_attention_output_proj(
        q, k, v, o_weight=o_weight, causal=causal, scale=scale,
    )


@cpu.register("gated_mlp")
def gated_mlp(x, *, gate_weight, up_weight, down_weight, activation="silu"):
    return ref_gated_mlp(
        x, gate_weight=gate_weight, up_weight=up_weight,
        down_weight=down_weight, activation=activation,
    )
