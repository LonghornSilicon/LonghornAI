"""Llama-style gated MLP block, composed from Phase-1 kernels.

Demonstrates kernel composition (RMSNorm -> gate/up projections -> SwiGLU ->
down projection) using only LonghornAI public kernels, so the model-level
accuracy harness (PLAN.md §5.3) can certify the composition end to end.
"""

from __future__ import annotations

import numpy as np

from ..kernels import gemm, rmsnorm, silu


def llama_mlp_block(x, w_gate, w_up, w_down, norm_weight=None, eps: float = 1e-6):
    """SwiGLU MLP with a pre-norm.

    x        : (tokens, hidden)
    w_gate   : (hidden, intermediate)
    w_up     : (hidden, intermediate)
    w_down   : (intermediate, hidden)
    """
    h = rmsnorm(x, weight=norm_weight, eps=eps)
    gate = silu(gemm(h, w_gate))
    up = gemm(h, w_up)
    fused = (gate.astype(np.float32) * up.astype(np.float32)).astype(x.dtype)
    return gemm(fused, w_down)
