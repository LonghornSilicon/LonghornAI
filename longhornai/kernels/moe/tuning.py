"""Auto-tuning configuration spaces for the MoE family."""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

ROUTER_TUNING_SPACE = TuningSpace(
    params={"vec_width": [4, 8, 16], "fuse_softmax": [True, False]}
)

TOP_K_TUNING_SPACE = TuningSpace(
    params={"algorithm": ["argsort", "argpartition", "bitonic"]}
)

DISPATCH_TUNING_SPACE = TuningSpace(
    params={
        "scatter_strategy": ["sort", "atomic", "histogram"],
        "vec_width": [4, 8, 16],
    }
)

COMBINE_TUNING_SPACE = TuningSpace(
    params={"reduction": ["serial", "tree"], "vec_width": [4, 8, 16]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "moe_router": ROUTER_TUNING_SPACE,
    "moe_top_k": TOP_K_TUNING_SPACE,
    "moe_dispatch": DISPATCH_TUNING_SPACE,
    "moe_combine": COMBINE_TUNING_SPACE,
}

__all__ = [
    "ROUTER_TUNING_SPACE",
    "TOP_K_TUNING_SPACE",
    "DISPATCH_TUNING_SPACE",
    "COMBINE_TUNING_SPACE",
    "TUNING_SPACES",
]
