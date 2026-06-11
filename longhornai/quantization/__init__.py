"""Quantization algorithms & packed formats (PLAN.md §3 Phase 4).

M1 provides the dtype/scale primitives the later AWQ/GPTQ/SmoothQuant kernels
build on: symmetric/affine quantize-dequantize and per-tensor / per-channel
scale derivation. The full algorithms (calibration, packed-format matmul) land
in the Phase-4 milestone (M5).
"""

from __future__ import annotations

from .primitives import dequantize, quantize, quantize_dequantize, scales_per_channel

__all__ = ["quantize", "dequantize", "quantize_dequantize", "scales_per_channel"]
