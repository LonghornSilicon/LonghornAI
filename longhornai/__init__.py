"""LonghornAI — canonical AI inference kernel library for Longhorn Silicon.

This package is the M1 reference implementation: Phase-1 foundational compute
kernels backed by a portable CPU (NumPy) reference backend, a backend-dispatch
runtime, a numerical validation harness, and a benchmarking harness.

The public surface is intentionally small and stable:

    >>> import longhornai as lh
    >>> import numpy as np
    >>> a = np.random.randn(64, 128).astype("float32")
    >>> b = np.random.randn(128, 32).astype("float32")
    >>> c = lh.gemm(a, b)            # dispatches to the active backend
    >>> c.shape
    (64, 32)

Kernels are dispatched through :mod:`longhornai.runtime`, so the same call
re-targets across backends (CPU today; sim / RTL / FPGA / silicon per PLAN.md)
without changing user code.
"""

from __future__ import annotations

from .runtime import (
    Backend,
    available_backends,
    dispatch,
    get_backend,
    set_default_backend,
    use_backend,
)
from .kernels import (
    activation_dequantize,
    activation_quantize,
    all_gather,
    all_reduce,
    all_to_all,
    attention_output_proj,
    batched_gemm,
    block_sparse_attention,
    dilated_attention,
    embedding_lookup,
    flash_attention_v1,
    flash_attention_v2,
    gated_mlp,
    gelu,
    gemm,
    gemm_w4a16,
    gemm_w8a8,
    gqa,
    grouped_gemm,
    kv_cache_alloc,
    kv_cache_alloc_int8,
    kv_cache_append,
    kv_cache_dequantize_gather,
    kv_cache_gather,
    kv_cache_quantize_append,
    layernorm,
    mha,
    moe_combine,
    moe_dispatch,
    moe_router,
    moe_top_k,
    mqa,
    multi_head_attention,
    paged_attention,
    paged_kv_alloc,
    paged_kv_append,
    reduce,
    reduce_scatter,
    rmsnorm,
    rmsnorm_qkv,
    rope,
    sdpa,
    silu,
    sliding_window_attention,
    softmax,
    tensor_contraction,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # runtime
    "Backend",
    "dispatch",
    "get_backend",
    "set_default_backend",
    "available_backends",
    "use_backend",
    # kernels (Phase 1)
    "gemm",
    "batched_gemm",
    "grouped_gemm",
    "tensor_contraction",
    "layernorm",
    "rmsnorm",
    "softmax",
    "gelu",
    "silu",
    "rope",
    "embedding_lookup",
    "reduce",
    # kernels (Phase 2 / M2)
    "sdpa",
    "flash_attention_v1",
    # kernels (Phase 2 / M3)
    "flash_attention_v2",
    "multi_head_attention",
    "mha",
    "mqa",
    "gqa",
    "kv_cache_alloc",
    "kv_cache_append",
    "kv_cache_gather",
    "kv_cache_alloc_int8",
    "kv_cache_quantize_append",
    "kv_cache_dequantize_gather",
    # kernels (Phase 2 / M4)
    "paged_kv_alloc",
    "paged_kv_append",
    "paged_attention",
    # kernels (Phase 4 / M5)
    "gemm_w8a8",
    "gemm_w4a16",
    "activation_quantize",
    "activation_dequantize",
    # kernels (Phase 6 / M6 — collectives)
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    # kernels (Phase 5 / M6 — MoE)
    "moe_router",
    "moe_top_k",
    "moe_dispatch",
    "moe_combine",
    # kernels (Phase 6 / M7 — sparse attention)
    "sliding_window_attention",
    "block_sparse_attention",
    "dilated_attention",
    # kernels (Phase 6 / M7 — fused)
    "rmsnorm_qkv",
    "attention_output_proj",
    "gated_mlp",
]
