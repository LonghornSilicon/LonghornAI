"""Auto-tuning configuration spaces for the KV-cache family.

Append/gather are I/O-bound layout shuffles — the meaningful perf knobs on
real hardware are vector width, head-major vs sequence-major layout, and (for
the INT8 path) whether dequant fuses into the gather pass or runs as a
separate kernel.
"""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

KV_APPEND_TUNING_SPACE = TuningSpace(
    params={"vec_width": [8, 16, 32], "layout": ["bhsd", "bshd"]}
)

KV_GATHER_TUNING_SPACE = TuningSpace(
    params={"vec_width": [8, 16, 32]}
)

KV_QUANTIZE_TUNING_SPACE = TuningSpace(
    params={"vec_width": [8, 16], "fuse_scale_compute": [True, False]}
)

KV_DEQUANTIZE_TUNING_SPACE = TuningSpace(
    params={"vec_width": [8, 16], "fuse_with_attn": [True, False]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "kv_cache_append": KV_APPEND_TUNING_SPACE,
    "kv_cache_gather": KV_GATHER_TUNING_SPACE,
    "kv_cache_quantize_append": KV_QUANTIZE_TUNING_SPACE,
    "kv_cache_dequantize_gather": KV_DEQUANTIZE_TUNING_SPACE,
}

__all__ = [
    "KV_APPEND_TUNING_SPACE",
    "KV_GATHER_TUNING_SPACE",
    "KV_QUANTIZE_TUNING_SPACE",
    "KV_DEQUANTIZE_TUNING_SPACE",
    "TUNING_SPACES",
]
