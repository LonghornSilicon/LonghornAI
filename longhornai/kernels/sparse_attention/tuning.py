"""Auto-tuning configuration spaces for the sparse-attention family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

SLIDING_WINDOW_TUNING_SPACE = TuningSpace(
    params={
        "block_q": [32, 64, 128],
        "block_kv": [32, 64, 128],
        "stages": [2, 3],
    }
)

BLOCK_SPARSE_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [32, 64, 128],
        "skip_threshold": [0.0, 0.5, 0.8],   # min block density to compute
    }
)

DILATED_TUNING_SPACE = TuningSpace(
    params={"vec_width": [4, 8, 16]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "sliding_window_attention": SLIDING_WINDOW_TUNING_SPACE,
    "block_sparse_attention": BLOCK_SPARSE_TUNING_SPACE,
    "dilated_attention": DILATED_TUNING_SPACE,
}

__all__ = [
    "SLIDING_WINDOW_TUNING_SPACE",
    "BLOCK_SPARSE_TUNING_SPACE",
    "DILATED_TUNING_SPACE",
    "TUNING_SPACES",
]
