"""Auto-tuning configuration spaces for the elementwise / sequence family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

SOFTMAX_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [128, 256, 512],
        "vec_width": [4, 8],
        "online": [True, False],
    }
)

ROPE_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [64, 128, 256],
        "layout": ["interleaved", "half"],
    }
)

EMBEDDING_TUNING_SPACE = TuningSpace(
    params={"vec_width": [4, 8, 16]}
)

REDUCE_TUNING_SPACE = TuningSpace(
    params={"block_size": [128, 256, 512], "vec_width": [4, 8, 16]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "softmax": SOFTMAX_TUNING_SPACE,
    "rope": ROPE_TUNING_SPACE,
    "embedding_lookup": EMBEDDING_TUNING_SPACE,
    "reduce": REDUCE_TUNING_SPACE,
}

__all__ = [
    "SOFTMAX_TUNING_SPACE",
    "ROPE_TUNING_SPACE",
    "EMBEDDING_TUNING_SPACE",
    "REDUCE_TUNING_SPACE",
    "TUNING_SPACES",
]
