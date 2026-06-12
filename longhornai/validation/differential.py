"""Differential testing: backend kernel vs. float64 golden reference.

Generates randomized inputs of a requested shape/dtype, runs both the active
backend implementation and the golden reference, and asserts agreement under the
per-dtype tolerance policy (PLAN.md §5.1). Reports the achieved max relative
error so callers can track headroom against the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import REFERENCES
from .tolerance import max_rel_error, tolerance_for


def _dtype_name(dtype) -> str:
    """Resolve a dtype label that may be a string (e.g. ``"bfloat16"``)."""
    if isinstance(dtype, str):
        return dtype
    return np.dtype(dtype).name


@dataclass
class DiffResult:
    """Outcome of one differential comparison."""

    op: str
    dtype: str
    passed: bool
    max_rel_error: float
    rtol: float
    atol: float

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.op:<20} {self.dtype:<9} "
            f"max_rel_err={self.max_rel_error:.2e} (rtol={self.rtol:.0e})"
        )


def differential_test(
    op: str,
    kernel_fn,
    *args,
    dtype=np.float32,
    **kwargs,
) -> DiffResult:
    """Compare ``kernel_fn(*args, **kwargs)`` against the golden for ``op``.

    The golden is looked up from :data:`REFERENCES`. ``dtype`` may be a NumPy
    dtype, a type, or a string label (``"bfloat16"`` is the emulated bf16
    carrier whose tolerance lives in the per-dtype policy).
    """
    if op not in REFERENCES:
        raise KeyError(f"no golden reference registered for op '{op}'")
    ref_fn = REFERENCES[op]

    actual = kernel_fn(*args, **kwargs)
    expected = ref_fn(*args, **kwargs)

    # grouped_gemm returns a list; concatenate flat for a single error figure.
    if isinstance(actual, list):
        actual = np.concatenate([np.asarray(a).ravel() for a in actual])
        expected = np.concatenate([np.asarray(e).ravel() for e in expected])

    name = _dtype_name(dtype)
    tol = tolerance_for(name)
    rel = max_rel_error(actual, expected)
    passed = bool(
        np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=tol.rtol,
            atol=tol.atol,
            equal_nan=True,
        )
    )
    return DiffResult(
        op=op,
        dtype=name,
        passed=passed,
        max_rel_error=rel,
        rtol=tol.rtol,
        atol=tol.atol,
    )
