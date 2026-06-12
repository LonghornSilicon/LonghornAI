"""bfloat16 emulation for testing.

NumPy has no native bf16 dtype, so we emulate by storing values as float32
with the mantissa truncated to bf16 precision (7 explicit bits). The
:func:`to_bf16` helper performs round-to-nearest-even truncation; the
resulting array is a fp32 carrier whose values exactly match what bf16
storage would produce.

The string label ``"bfloat16"`` is the dtype identifier the rest of the
validation harness uses to look up the bf16 tolerance from
:mod:`longhornai.validation.tolerance` (PLAN.md §5.1 numerical-tolerance
policy). This decouples the carrier dtype from the *contractual* dtype.

Real silicon backends will replace this emulation with native bf16 storage;
the same tolerance policy applies unchanged.
"""

from __future__ import annotations

import numpy as np

BF16_DTYPE_NAME = "bfloat16"


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round ``x`` to bfloat16 precision; return as float32 (carrier dtype).

    Round-to-nearest-even on the low 16 bits of the IEEE-754 fp32
    representation: bf16 = fp32 with the lower 16 mantissa bits zeroed,
    after a parity-aware rounding bump. Special values (NaN/±Inf/±0) are
    preserved.
    """
    x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = x32.view(np.uint32).copy()
    # Round-to-nearest-even: add 0x7FFF (the rounding bias) plus the LSB of
    # the surviving bits, so ties round to even.
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    # NaN survives the bump (high-mantissa bits stay set); zero out low 16.
    rounded &= 0xFFFF0000
    return rounded.view(np.float32).copy()


def is_bf16_label(dtype) -> bool:
    """True if ``dtype`` is the string label for the emulated bf16 carrier."""
    if isinstance(dtype, str):
        return dtype == BF16_DTYPE_NAME
    try:
        return np.dtype(dtype).name == BF16_DTYPE_NAME
    except TypeError:
        return False


__all__ = ["BF16_DTYPE_NAME", "to_bf16", "is_bf16_label"]
