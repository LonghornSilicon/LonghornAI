"""Kernel codegen & auto-tuning (PLAN.md §6.1 auto-tuning framework).

M1 ships the auto-tuner *interface* and an exhaustive search driver over a
config space; target-specialized codegen lowering hooks land with the sim/RTL
backends. This package is the seed crystal for the Longhorn Compiler Stack
(PLAN.md §10.2).
"""

from __future__ import annotations

from .autotune import TuningResult, TuningSpace, autotune
from .autotune_cache import TuneCache, TuneKey, autotune_cached, shape_signature

__all__ = [
    "TuningSpace",
    "TuningResult",
    "autotune",
    "TuneCache",
    "TuneKey",
    "autotune_cached",
    "shape_signature",
]
