"""Per-dtype numerical-tolerance policy.

The single source of truth for "do two tensors agree?" across the whole repo
(PLAN.md §5.1 numerical validation). Tolerances are deliberately conservative
and dtype-aware: FP32 is held tight, reduced-precision types are allowed the
slack their mantissa demands. Any kernel, anywhere, that needs to compare
against a golden uses these numbers — not ad-hoc per-test constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """Relative and absolute tolerance for a comparison."""

    rtol: float
    atol: float


# bf16 is represented as float32 in NumPy (no native dtype); we key it by name.
_POLICY: dict[str, Tolerance] = {
    "float64": Tolerance(rtol=1e-12, atol=1e-12),
    "float32": Tolerance(rtol=1e-5, atol=1e-6),
    "float16": Tolerance(rtol=1e-2, atol=1e-2),
    "bfloat16": Tolerance(rtol=2e-2, atol=2e-2),
}


def tolerance_for(dtype: object) -> Tolerance:
    """Return the policy tolerance for ``dtype`` (numpy dtype, type, or name)."""
    if isinstance(dtype, str):
        name = dtype
    else:
        name = np.dtype(dtype).name
    if name not in _POLICY:
        raise KeyError(f"no tolerance policy for dtype '{name}'; known: {sorted(_POLICY)}")
    return _POLICY[name]


def max_rel_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Max relative error of ``actual`` vs ``reference`` (float64, NaN-safe)."""
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    denom = np.maximum(np.abs(r), 1e-30)
    return float(np.max(np.abs(a - r) / denom))


def assert_close(
    actual: np.ndarray,
    reference: np.ndarray,
    dtype: object,
    *,
    name: str = "tensor",
) -> None:
    """Raise ``AssertionError`` if ``actual`` and ``reference`` disagree under policy."""
    tol = tolerance_for(dtype)
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {r.shape}")
    if not np.allclose(a, r, rtol=tol.rtol, atol=tol.atol, equal_nan=True):
        rel = max_rel_error(a, r)
        abs_err = float(np.max(np.abs(a - r)))
        raise AssertionError(
            f"{name}: exceeds {dtype} tolerance "
            f"(rtol={tol.rtol}, atol={tol.atol}); "
            f"max_abs_err={abs_err:.3e}, max_rel_err={rel:.3e}"
        )
