"""Auto-tuning configuration spaces for the activation family.

Activations are pure elementwise — the search space is small: vector width
(loads/stores per thread) and which approximation to take when applicable.
"""

from __future__ import annotations

from typing import Dict

from ...compiler import TuningSpace

GELU_TUNING_SPACE = TuningSpace(
    params={
        "vec_width": [4, 8, 16],
        "approximate": ["none", "tanh"],
    }
)

SILU_TUNING_SPACE = TuningSpace(
    params={"vec_width": [4, 8, 16]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "gelu": GELU_TUNING_SPACE,
    "silu": SILU_TUNING_SPACE,
}

__all__ = ["GELU_TUNING_SPACE", "SILU_TUNING_SPACE", "TUNING_SPACES"]
