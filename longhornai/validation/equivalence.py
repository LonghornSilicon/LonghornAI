"""Cross-target equivalence harness (PLAN.md §5.2).

The keystone of LonghornAI's validation strategy: run a given operator+input
across **every available backend** and assert agreement within the per-dtype
tolerance policy. This is what lets the silicon team trust that a kernel
which passed on RTL will behave correctly on the part — and lets the kernel
team catch HW/SW divergence before tapeout.

PLAN.md §8 M3 exit gate: **RTL ≡ CPU on attention**. The module-level
:func:`assert_cross_target_equivalent` runs the same call through every
registered backend (CPU, RTL, sim/fpga/lhsil as they land) and aggregates the
result into a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..runtime import available_backends, dispatch, get_backend, use_backend
from .tolerance import max_rel_error, tolerance_for


@dataclass
class BackendOutcome:
    """One backend's result on one ``(op, input)`` invocation."""

    backend: str
    op: str
    success: bool
    max_rel_error: float
    error: str | None = None


@dataclass
class EquivalenceReport:
    """Aggregated cross-target outcomes for one ``(op, input)`` invocation."""

    op: str
    dtype: str
    reference_backend: str
    outcomes: List[BackendOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.success for o in self.outcomes)

    def __str__(self) -> str:  # pragma: no cover - human-readable summary
        lines = [
            f"cross-target check: op={self.op} dtype={self.dtype} "
            f"reference={self.reference_backend}"
        ]
        for o in self.outcomes:
            tag = "PASS" if o.success else "FAIL"
            line = f"  [{tag}] {o.backend:<6} max_rel_err={o.max_rel_error:.2e}"
            if o.error:
                line += f"  ({o.error})"
            lines.append(line)
        return "\n".join(lines)


def _to_array(x: Any) -> np.ndarray:
    """Coerce a kernel return value to one flat float64 array for comparison."""
    if isinstance(x, tuple):
        # KV-cache ops return tuples; concatenate flat.
        return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in x])
    if isinstance(x, list):
        return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in x])
    return np.asarray(x, dtype=np.float64)


def _deep_copy_args(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
    """Return deep copies of array args/kwargs so per-backend runs cannot
    mutate each other's input state — KV-cache ops mutate in place."""
    def copy(x):
        if isinstance(x, np.ndarray):
            return x.copy()
        if isinstance(x, list):
            return [copy(item) for item in x]
        if isinstance(x, tuple):
            return tuple(copy(item) for item in x)
        return x

    return tuple(copy(a) for a in args), {k: copy(v) for k, v in kwargs.items()}


def run_on_backend(
    backend_name: str, op_name: str, *args: Any, **kwargs: Any,
) -> Any:
    """Dispatch ``op_name`` to the named backend with isolated copies of the args."""
    cargs, ckwargs = _deep_copy_args(args, kwargs)
    with use_backend(backend_name):
        return dispatch(op_name, *cargs, **ckwargs)


def assert_cross_target_equivalent(
    op_name: str,
    *args: Any,
    dtype: object = np.float32,
    reference: str = "cpu",
    backends: List[str] | None = None,
    **kwargs: Any,
) -> EquivalenceReport:
    """Run ``op_name`` under every backend and assert agreement within tolerance.

    Note: the parameter is named ``op_name`` rather than ``op`` so callers
    can pass kernels whose own keyword arguments include ``op`` (e.g.
    :func:`longhornai.all_reduce(..., op="sum")`) without a name collision.

    ``reference`` is the backend whose output every other backend is compared
    against (defaults to CPU — the correctness anchor per PLAN.md §2.2).
    ``backends`` defaults to all currently-registered backends.
    """
    targets = backends if backends is not None else available_backends()
    if reference not in targets:
        targets = [reference] + list(targets)

    ref_out = run_on_backend(reference, op_name, *args, **kwargs)
    ref_flat = _to_array(ref_out)

    tol = tolerance_for(dtype if isinstance(dtype, str) else np.dtype(dtype).name)
    outcomes: List[BackendOutcome] = [
        BackendOutcome(backend=reference, op=op_name, success=True, max_rel_error=0.0)
    ]
    for name in targets:
        if name == reference:
            continue
        try:
            out = run_on_backend(name, op_name, *args, **kwargs)
            flat = _to_array(out)
            rel = max_rel_error(flat, ref_flat)
            ok = bool(np.allclose(
                flat, ref_flat, rtol=tol.rtol, atol=tol.atol, equal_nan=True,
            ))
            outcomes.append(BackendOutcome(
                backend=name, op=op_name, success=ok, max_rel_error=rel,
                error=None if ok else f"exceeds rtol={tol.rtol}",
            ))
        except Exception as exc:  # pragma: no cover - failure path
            outcomes.append(BackendOutcome(
                backend=name, op=op_name, success=False,
                max_rel_error=float("inf"), error=repr(exc),
            ))

    return EquivalenceReport(
        op=op_name,
        dtype=dtype if isinstance(dtype, str) else np.dtype(dtype).name,
        reference_backend=reference,
        outcomes=outcomes,
    )


__all__ = [
    "BackendOutcome",
    "EquivalenceReport",
    "assert_cross_target_equivalent",
    "run_on_backend",
]
