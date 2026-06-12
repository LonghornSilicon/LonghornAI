"""Auto-tuning configuration spaces for the quantized-GEMM family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

GEMM_W8A8_TUNING_SPACE = TuningSpace(
    params={
        "tile_m": [64, 128, 256],
        "tile_n": [64, 128, 256],
        "tile_k": [32, 64],
        "split_k": [1, 2, 4],
    }
)

GEMM_W4A16_TUNING_SPACE = TuningSpace(
    params={
        "tile_m": [16, 32, 64, 128],
        "tile_n": [64, 128, 256],
        "group_size": [32, 64, 128],
        "fuse_dequant": [True, False],
    }
)

ACTIVATION_QUANTIZE_TUNING_SPACE = TuningSpace(
    params={
        "vec_width": [8, 16, 32],
        "fuse_into_prologue": [True, False],
    }
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "gemm_w8a8": GEMM_W8A8_TUNING_SPACE,
    "gemm_w4a16": GEMM_W4A16_TUNING_SPACE,
    "activation_quantize": ACTIVATION_QUANTIZE_TUNING_SPACE,
}

__all__ = [
    "GEMM_W8A8_TUNING_SPACE",
    "GEMM_W4A16_TUNING_SPACE",
    "ACTIVATION_QUANTIZE_TUNING_SPACE",
    "TUNING_SPACES",
]
