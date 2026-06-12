"""Auto-tuning configuration spaces for the attention family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

SDPA_TUNING_SPACE = TuningSpace(
    params={"fuse_softmax": [True, False]}
)

FLASH_V1_TUNING_SPACE = TuningSpace(
    params={
        "block_q": [32, 64, 128],
        "block_kv": [32, 64, 128],
        "stages": [2, 3],
    }
)

# v2 has the same tile space as v1 plus a "warp_split" knob — it's the
# v2-specific work-partitioning dial that real GPUs care about and is the
# whole reason v2 exists. The CPU reference treats it as ABI-only.
FLASH_V2_TUNING_SPACE = TuningSpace(
    params={
        "block_q": [64, 128, 256],
        "block_kv": [32, 64, 128],
        "stages": [2, 3],
        "warp_split": ["q", "kv"],
    }
)

MHA_TUNING_SPACE = TuningSpace(
    params={
        "attn_impl": ["sdpa", "flash_v1", "flash_v2"],
        "fuse_qkv_proj": [True, False],
        "fuse_o_proj": [True, False],
    }
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "sdpa": SDPA_TUNING_SPACE,
    "flash_attention_v1": FLASH_V1_TUNING_SPACE,
    "flash_attention_v2": FLASH_V2_TUNING_SPACE,
    "multi_head_attention": MHA_TUNING_SPACE,
}

__all__ = [
    "SDPA_TUNING_SPACE",
    "FLASH_V1_TUNING_SPACE",
    "FLASH_V2_TUNING_SPACE",
    "MHA_TUNING_SPACE",
    "TUNING_SPACES",
]
