"""Quantization algorithms & packed formats (PLAN.md §3 Phase 4 / M5).

M1 shipped the symmetric quant/dequant primitives. M5 extends:

* asymmetric (zero-point) and groupwise quantization,
* INT4 packing (two nibbles per byte),
* SmoothQuant / AWQ / GPTQ calibration algorithms (in :mod:`.calibration`).

The W8A8 / W4A16 GEMM kernels live in :mod:`longhornai.kernels.quant`; this
package owns the *algorithmic* layer (how to derive quantized weights) and
the *format* primitives (pack/unpack).
"""

from __future__ import annotations

from .calibration import (
    AWQResult,
    GPTQResult,
    SmoothQuantResult,
    awq_calibrate,
    gptq_calibrate,
    smooth_quant_calibrate,
)
from .primitives import (
    QuantParams,
    dequantize,
    dequantize_q,
    pack_int4,
    quantize,
    quantize_asymmetric,
    quantize_dequantize,
    quantize_groupwise,
    scales_per_channel,
    unpack_int4,
)

__all__ = [
    # primitives
    "QuantParams",
    "quantize",
    "dequantize",
    "quantize_dequantize",
    "scales_per_channel",
    "quantize_asymmetric",
    "quantize_groupwise",
    "dequantize_q",
    "pack_int4",
    "unpack_int4",
    # calibration (M5)
    "smooth_quant_calibrate",
    "SmoothQuantResult",
    "awq_calibrate",
    "AWQResult",
    "gptq_calibrate",
    "GPTQResult",
]
