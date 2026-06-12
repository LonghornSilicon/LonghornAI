"""Auto-tuning configuration spaces for the normalization family.

These kernels are reduction-heavy and bandwidth-bound; the meaningful knobs
are block size (rows per warp/block), vector width (loads per thread), and
whether the affine epilogue is fused into the same launch.
"""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

LAYERNORM_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [128, 256, 512],
        "vec_width": [4, 8, 16],
        "fuse_affine": [True, False],
    }
)

RMSNORM_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [128, 256, 512],
        "vec_width": [4, 8, 16],
    }
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "layernorm": LAYERNORM_TUNING_SPACE,
    "rmsnorm": RMSNORM_TUNING_SPACE,
}

__all__ = ["LAYERNORM_TUNING_SPACE", "RMSNORM_TUNING_SPACE", "TUNING_SPACES"]
