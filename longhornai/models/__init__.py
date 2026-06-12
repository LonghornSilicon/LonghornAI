"""End-to-end model graphs assembled from LonghornAI kernels (PLAN.md §7).

M1 shipped a single Llama-style SwiGLU MLP block to prove that kernel
*composition* is correct. M2 lands the full Llama forward —
embedding → N decoder layers (pre-norm attention + pre-norm SwiGLU MLP) →
final norm → LM head — composed entirely from registered LonghornAI
kernels, certifying "Llama FP16 forward correct (CPU/sim)".

M3 adds Mistral and Qwen forward configs (PLAN.md §8 M3 — "Mistral + Qwen
FP16 correct"). Both share Llama's architecture; the differences are
hyperparameters (Mistral: 4:1 GQA; Qwen: ``rope_theta = 1e6``) so they reuse
:func:`llama_forward` with their own configs.
"""

from __future__ import annotations

from .deepseek import (
    DeepSeekConfig,
    DeepSeekExpertWeights,
    DeepSeekLayerWeights,
    DeepSeekWeights,
    deepseek_decoder_layer,
    deepseek_forward,
    deepseek_toy_config,
    deepseek_v2_config,
    init_deepseek_weights,
)
from .gemma import gemma_2b_config, gemma_toy_config
from .llama import (
    LlamaConfig,
    LlamaLayerWeights,
    LlamaWeights,
    PagedKVState,
    alloc_paged_kv_state,
    init_random_weights,
    llama_decode_step,
    llama_decoder_layer,
    llama_forward,
    llama_prefill,
)
from .llama_block import llama_mlp_block
from .llama_quantized import (
    LlamaLayerWeightsQ,
    LlamaWeightsQ,
    llama_forward_quantized,
    quantize_weights_int4,
)
from .mistral import mistral_7b_config, mistral_toy_config
from .mixtral import (
    MixtralConfig,
    MixtralExpertWeights,
    MixtralLayerWeights,
    MixtralWeights,
    init_mixtral_weights,
    mixtral_8x7b_config,
    mixtral_decoder_layer,
    mixtral_forward,
    mixtral_toy_config,
)
from .phi import phi3_medium_config, phi3_mini_config, phi_toy_config
from .qwen import qwen2_5_7b_config, qwen_toy_config

__all__ = [
    "llama_mlp_block",
    "LlamaConfig",
    "LlamaLayerWeights",
    "LlamaWeights",
    "PagedKVState",
    "init_random_weights",
    "alloc_paged_kv_state",
    "llama_decoder_layer",
    "llama_forward",
    "llama_prefill",
    "llama_decode_step",
    "mistral_7b_config",
    "mistral_toy_config",
    "qwen2_5_7b_config",
    "qwen_toy_config",
    "gemma_2b_config",
    "gemma_toy_config",
    "phi3_mini_config",
    "phi3_medium_config",
    "phi_toy_config",
    # M5 — quantized model
    "LlamaWeightsQ",
    "LlamaLayerWeightsQ",
    "quantize_weights_int4",
    "llama_forward_quantized",
    # M6 — Mixtral (sparse MoE)
    "MixtralConfig",
    "MixtralExpertWeights",
    "MixtralLayerWeights",
    "MixtralWeights",
    "init_mixtral_weights",
    "mixtral_decoder_layer",
    "mixtral_forward",
    "mixtral_8x7b_config",
    "mixtral_toy_config",
    # M7 — DeepSeek (MLA + fine-grained MoE)
    "DeepSeekConfig",
    "DeepSeekExpertWeights",
    "DeepSeekLayerWeights",
    "DeepSeekWeights",
    "init_deepseek_weights",
    "deepseek_decoder_layer",
    "deepseek_forward",
    "deepseek_v2_config",
    "deepseek_toy_config",
]
