"""LonghornAI kernel families.

These are the stable public kernel entry points. Each is a thin front-end that
dispatches through :func:`longhornai.runtime.dispatch` to the active backend, so
the same call re-targets across CPU / sim / RTL / FPGA / silicon.

Every kernel has a matching float64 golden in
``kernels/<family>/reference.py`` and a cost model in
:mod:`longhornai.benchmarks.cost` used by the benchmark harness.
"""

from __future__ import annotations

from .activation import gelu, silu
from .attention import (
    flash_attention_v1,
    flash_attention_v2,
    gqa,
    mha,
    mqa,
    multi_head_attention,
    sdpa,
)
from .elementwise import embedding_lookup, reduce, rope, softmax
from .gemm import batched_gemm, gemm, grouped_gemm, tensor_contraction
from .kv_cache import (
    kv_cache_alloc,
    kv_cache_alloc_int8,
    kv_cache_append,
    kv_cache_dequantize_gather,
    kv_cache_gather,
    kv_cache_quantize_append,
)
from .normalization import layernorm, rmsnorm
from .paged_attention import paged_attention, paged_kv_alloc, paged_kv_append
from .quant import (
    activation_dequantize,
    activation_quantize,
    gemm_w4a16,
    gemm_w8a8,
)

__all__ = [
    # GEMM family
    "gemm",
    "batched_gemm",
    "grouped_gemm",
    "tensor_contraction",
    # normalization family
    "layernorm",
    "rmsnorm",
    # activation family
    "gelu",
    "silu",
    # elementwise / sequence family
    "softmax",
    "rope",
    "embedding_lookup",
    "reduce",
    # attention family (M2)
    "sdpa",
    "flash_attention_v1",
    # attention family (M3)
    "flash_attention_v2",
    "multi_head_attention",
    "mha",
    "mqa",
    "gqa",
    # kv-cache family (M3)
    "kv_cache_alloc",
    "kv_cache_append",
    "kv_cache_gather",
    "kv_cache_alloc_int8",
    "kv_cache_quantize_append",
    "kv_cache_dequantize_gather",
    # paged-attention family (M4)
    "paged_kv_alloc",
    "paged_kv_append",
    "paged_attention",
    # quant family (M5)
    "gemm_w8a8",
    "gemm_w4a16",
    "activation_quantize",
    "activation_dequantize",
]
