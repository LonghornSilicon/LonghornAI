"""Auto-tuning configuration spaces for the paged-attention family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

PAGED_KV_APPEND_TUNING_SPACE = TuningSpace(
    params={"vec_width": [4, 8, 16], "scatter": ["per_token", "per_block"]}
)

PAGED_ATTENTION_TUNING_SPACE = TuningSpace(
    params={
        "block_size": [16, 32, 64, 128],
        "split_kv": [True, False],   # v2-style split-KV reduction across blocks
        "warps_per_block": [2, 4, 8],
    }
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "paged_kv_append": PAGED_KV_APPEND_TUNING_SPACE,
    "paged_attention": PAGED_ATTENTION_TUNING_SPACE,
}

__all__ = [
    "PAGED_KV_APPEND_TUNING_SPACE",
    "PAGED_ATTENTION_TUNING_SPACE",
    "TUNING_SPACES",
]
