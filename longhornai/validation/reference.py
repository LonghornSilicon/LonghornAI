"""Float64 golden references — aggregator.

Each kernel family owns its references in ``kernels/<family>/reference.py``.
This module aggregates them into the :data:`REFERENCES` registry consumed by
:func:`longhornai.validation.differential_test`.

Re-exports keep the historical ``ref_*`` import surface stable so existing
callers continue to work.
"""

from __future__ import annotations

from typing import Callable, Dict

from ..kernels.activation.reference import ref_gelu, ref_gelu_exact, ref_silu
from ..kernels.attention.reference import (
    ref_flash_attention_v1,
    ref_flash_attention_v2,
    ref_multi_head_attention,
    ref_sdpa,
)
from ..kernels.collectives.reference import (
    ref_all_gather,
    ref_all_reduce,
    ref_all_to_all,
    ref_reduce_scatter,
)
from ..kernels.fused.reference import (
    ref_attention_output_proj,
    ref_gated_mlp,
    ref_rmsnorm_qkv,
)
from ..kernels.kv_cache.reference import (
    ref_kv_cache_append,
    ref_kv_cache_dequantize_gather,
    ref_kv_cache_gather,
    ref_kv_cache_quantize_append,
)
from ..kernels.moe.reference import (
    ref_moe_combine,
    ref_moe_dispatch,
    ref_moe_router,
    ref_moe_top_k,
)
from ..kernels.paged_attention.reference import (
    ref_paged_attention,
    ref_paged_kv_append,
)
from ..kernels.quant.reference import (
    ref_gemm_w4a16,
    ref_gemm_w8a8,
)
from ..kernels.sparse_attention.reference import (
    ref_block_sparse_attention,
    ref_dilated_attention,
    ref_sliding_window_attention,
)
from ..kernels.elementwise.reference import (
    ref_embedding_lookup,
    ref_reduce,
    ref_rope,
    ref_softmax,
)
from ..kernels.gemm.reference import (
    ref_batched_gemm,
    ref_gemm,
    ref_grouped_gemm,
    ref_tensor_contraction,
)
from ..kernels.normalization.reference import ref_layernorm, ref_rmsnorm

REFERENCES: Dict[str, Callable] = {
    "gemm": ref_gemm,
    "batched_gemm": ref_batched_gemm,
    "grouped_gemm": ref_grouped_gemm,
    "tensor_contraction": ref_tensor_contraction,
    "layernorm": ref_layernorm,
    "rmsnorm": ref_rmsnorm,
    "gelu": ref_gelu_exact,
    "gelu_tanh": lambda x: ref_gelu(x, approximate="tanh"),
    "silu": ref_silu,
    "softmax": ref_softmax,
    "rope": ref_rope,
    "embedding_lookup": ref_embedding_lookup,
    "reduce": ref_reduce,
    "sdpa": ref_sdpa,
    "flash_attention_v1": ref_flash_attention_v1,
    "flash_attention_v2": ref_flash_attention_v2,
    "multi_head_attention": ref_multi_head_attention,
    "kv_cache_append": ref_kv_cache_append,
    "kv_cache_gather": ref_kv_cache_gather,
    "kv_cache_quantize_append": ref_kv_cache_quantize_append,
    "kv_cache_dequantize_gather": ref_kv_cache_dequantize_gather,
    "paged_kv_append": ref_paged_kv_append,
    "paged_attention": ref_paged_attention,
    "gemm_w8a8": ref_gemm_w8a8,
    "gemm_w4a16": ref_gemm_w4a16,
    "all_reduce": ref_all_reduce,
    "all_gather": ref_all_gather,
    "reduce_scatter": ref_reduce_scatter,
    "all_to_all": ref_all_to_all,
    "moe_router": ref_moe_router,
    "moe_top_k": ref_moe_top_k,
    "moe_dispatch": ref_moe_dispatch,
    "moe_combine": ref_moe_combine,
    "sliding_window_attention": ref_sliding_window_attention,
    "block_sparse_attention": ref_block_sparse_attention,
    "dilated_attention": ref_dilated_attention,
    "rmsnorm_qkv": ref_rmsnorm_qkv,
    "attention_output_proj": ref_attention_output_proj,
    "gated_mlp": ref_gated_mlp,
}

__all__ = [
    "REFERENCES",
    "ref_gemm",
    "ref_batched_gemm",
    "ref_grouped_gemm",
    "ref_tensor_contraction",
    "ref_layernorm",
    "ref_rmsnorm",
    "ref_gelu",
    "ref_gelu_exact",
    "ref_silu",
    "ref_softmax",
    "ref_rope",
    "ref_embedding_lookup",
    "ref_reduce",
    "ref_sdpa",
    "ref_flash_attention_v1",
    "ref_flash_attention_v2",
    "ref_multi_head_attention",
    "ref_kv_cache_append",
    "ref_kv_cache_gather",
    "ref_kv_cache_quantize_append",
    "ref_kv_cache_dequantize_gather",
    "ref_paged_kv_append",
    "ref_paged_attention",
    "ref_gemm_w8a8",
    "ref_gemm_w4a16",
    "ref_all_reduce",
    "ref_all_gather",
    "ref_reduce_scatter",
    "ref_all_to_all",
    "ref_moe_router",
    "ref_moe_top_k",
    "ref_moe_dispatch",
    "ref_moe_combine",
    "ref_sliding_window_attention",
    "ref_block_sparse_attention",
    "ref_dilated_attention",
    "ref_rmsnorm_qkv",
    "ref_attention_output_proj",
    "ref_gated_mlp",
]
