"""Calibration algorithms — SmoothQuant, AWQ, GPTQ.

These are *algorithms*, not runtime kernels: each takes a weight tensor and
calibration activations and produces calibrated quantization parameters
(plus possibly a transformed weight). The runtime path is the W8A8 / W4A16
GEMM kernels in :mod:`longhornai.kernels.quant`.

PLAN.md §3 Phase 4 calls for SmoothQuant, AWQ, and GPTQ; PLAN.md §8 M5
exit gate is "INT4/INT8 accuracy targets met at measured speedup". The
reference implementations here are deliberately small — readable, correct,
and tractable on toy weights. Production calibration runs the same math at
larger scale.
"""

from __future__ import annotations

from .awq import AWQResult, awq_calibrate
from .gptq import GPTQResult, gptq_calibrate
from .smooth_quant import SmoothQuantResult, smooth_quant_calibrate

__all__ = [
    "smooth_quant_calibrate",
    "SmoothQuantResult",
    "awq_calibrate",
    "AWQResult",
    "gptq_calibrate",
    "GPTQResult",
]
