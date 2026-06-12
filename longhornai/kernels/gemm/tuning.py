"""Auto-tuning configuration spaces for the GEMM family.

Phase-1 exit gate (PLAN.md §3) calls for "tuning framework operational for
GEMM". This module declares the search space (tile sizes, pipeline depth,
double-buffering) and a turn-key :func:`autotune_gemm` driver that hands a
factory to the generic search engine in :mod:`longhornai.compiler`.

On the CPU reference backend the tile knobs are ABI placeholders — every
config produces the same NumPy callable. They become load-bearing when the
sim/RTL/FPGA/silicon backends consume them through dispatch (PLAN.md §2.2).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...compiler import TuningResult, TuningSpace, autotune
from . import gemm as _gemm_kernel

GEMM_TUNING_SPACE = TuningSpace(
    params={
        "tile_m": [64, 128, 256],
        "tile_n": [64, 128, 256],
        "tile_k": [16, 32, 64],
        "stages": [2, 3],
    }
)

BATCHED_GEMM_TUNING_SPACE = GEMM_TUNING_SPACE
GROUPED_GEMM_TUNING_SPACE = TuningSpace(
    params={"tile_m": [32, 64, 128], "tile_k": [16, 32], "stages": [2]}
)

TUNING_SPACES: Dict[str, TuningSpace] = {
    "gemm": GEMM_TUNING_SPACE,
    "batched_gemm": BATCHED_GEMM_TUNING_SPACE,
    "grouped_gemm": GROUPED_GEMM_TUNING_SPACE,
}


def autotune_gemm(
    M: int,
    K: int,
    N: int,
    *,
    dtype=np.float16,
    warmup: int = 1,
    trials: int = 5,
    seed: int = 0,
) -> TuningResult:
    """Search ``GEMM_TUNING_SPACE`` at ``(M, K, N, dtype)`` and return the best config."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(dtype)
    b = rng.standard_normal((K, N)).astype(dtype)

    def factory(**_config: Any):
        # Tile/stage knobs are consumed by hardware-representative backends;
        # the CPU reference backend ignores them.
        return lambda: _gemm_kernel(a, b)

    return autotune(factory, GEMM_TUNING_SPACE, warmup=warmup, trials=trials)


__all__ = [
    "GEMM_TUNING_SPACE",
    "BATCHED_GEMM_TUNING_SPACE",
    "GROUPED_GEMM_TUNING_SPACE",
    "TUNING_SPACES",
    "autotune_gemm",
]
