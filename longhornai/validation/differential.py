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

    The golden is looked up from :data:`REFERENCES`. Array args are passed to the
    reference in float64; the kernel receives them in the requested ``dtype``.
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

    tol = tolerance_for(np.dtype(dtype).name)
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
        dtype=np.dtype(dtype).name,
        passed=passed,
        max_rel_error=rel,
        rtol=tol.rtol,
        atol=tol.atol,
    )
