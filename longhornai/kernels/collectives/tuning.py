"""Auto-tuning configuration spaces for the collectives family.

Real silicon backends pick algorithms based on message size and topology.
On the CPU shim these knobs are ABI-only.
"""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

ALL_REDUCE_TUNING_SPACE = TuningSpace(
    params={
        "algorithm": ["ring", "tree", "hierarchical"],
        "chunk_size_kb": [256, 1024, 4096],
    }
)

ALL_GATHER_TUNING_SPACE = TuningSpace(
    params={"algorithm": ["ring", "direct"]}
)

REDUCE_SCATTER_TUNING_SPACE = TuningSpace(
    params={"algorithm": ["ring", "tree"]}
)

ALL_TO_ALL_TUNING_SPACE = TuningSpace(
    params={"algorithm": ["pairwise", "ring", "hierarchical"]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "all_reduce": ALL_REDUCE_TUNING_SPACE,
    "all_gather": ALL_GATHER_TUNING_SPACE,
    "reduce_scatter": REDUCE_SCATTER_TUNING_SPACE,
    "all_to_all": ALL_TO_ALL_TUNING_SPACE,
}

__all__ = [
    "ALL_REDUCE_TUNING_SPACE",
    "ALL_GATHER_TUNING_SPACE",
    "REDUCE_SCATTER_TUNING_SPACE",
    "ALL_TO_ALL_TUNING_SPACE",
    "TUNING_SPACES",
]
