"""Auto-tuning configuration spaces for the fused-kernel family.

Fused kernels expose more knobs than their components — the loop ordering
of sub-ops, register allocation across stages, and which sub-op fuses
into the GEMM epilogue.
"""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

RMSNORM_QKV_TUNING_SPACE = TuningSpace(
    params={
        "tile_m": [64, 128, 256],
        "stages": [2, 3],
        "fuse_norm_into_qkv_prologue": [True, False],
    }
)

ATTENTION_OUTPUT_PROJ_TUNING_SPACE = TuningSpace(
    params={
        "block_q": [64, 128, 256],
        "block_kv": [32, 64, 128],
        "fuse_oproj_into_attn_epilogue": [True, False],
    }
)

GATED_MLP_TUNING_SPACE = TuningSpace(
    params={
        "tile_m": [64, 128, 256],
        "tile_n": [128, 256, 512],
        "stages": [2, 3],
        "fuse_act_mul_into_up_epilogue": [True, False],
    }
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "rmsnorm_qkv": RMSNORM_QKV_TUNING_SPACE,
    "attention_output_proj": ATTENTION_OUTPUT_PROJ_TUNING_SPACE,
    "gated_mlp": GATED_MLP_TUNING_SPACE,
}

__all__ = [
    "RMSNORM_QKV_TUNING_SPACE",
    "ATTENTION_OUTPUT_PROJ_TUNING_SPACE",
    "GATED_MLP_TUNING_SPACE",
    "TUNING_SPACES",
]
