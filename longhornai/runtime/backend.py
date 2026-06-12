"""Backend registry and operator dispatch.

A :class:`Backend` is a named bundle of operator implementations for one
execution target. Backends register themselves into a process-global registry;
kernels call :func:`dispatch` which routes ``op_name`` to the active backend's
implementation. This indirection is the seam that lets a single kernel front-end
re-target across CPU / sim / RTL / FPGA / silicon (PLAN.md §2.2).
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Dict, Iterator, List

OpImpl = Callable[..., Any]


class Backend:
    """A named collection of operator implementations for one target.

    Parameters
    ----------
    name:
        Stable identifier (e.g. ``"cpu"``, ``"sim"``, ``"lhsil"``).
    description:
        Human-readable summary, surfaced by the CLI and reports.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._ops: Dict[str, OpImpl] = {}

    def register(self, op_name: str) -> Callable[[OpImpl], OpImpl]:
        """Decorator registering ``fn`` as this backend's impl of ``op_name``."""

        def _decorator(fn: OpImpl) -> OpImpl:
            if op_name in self._ops:
                raise ValueError(f"backend '{self.name}' already implements '{op_name}'")
            self._ops[op_name] = fn
            return fn

        return _decorator

    def implements(self, op_name: str) -> bool:
        return op_name in self._ops

    def get(self, op_name: str) -> OpImpl:
        try:
            return self._ops[op_name]
        except KeyError as exc:
            raise NotImplementedError(
                f"backend '{self.name}' does not implement operator '{op_name}'"
            ) from exc

    def ops(self) -> List[str]:
        return sorted(self._ops)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Backend(name={self.name!r}, ops={len(self._ops)})"


# --- process-global registry -------------------------------------------------

_REGISTRY: Dict[str, Backend] = {}
_state = threading.local()


def register_backend(backend: Backend, *, default: bool = False) -> Backend:
    """Add ``backend`` to the registry; optionally make it the default target."""
    _REGISTRY[backend.name] = backend
    if default or _global_default() is None:
        _set_global_default(backend.name)
    return backend


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def get_backend(name: str | None = None) -> Backend:
    """Return the named backend, or the currently active one if ``name`` is None."""
    if name is None:
        name = _active_name()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown backend '{name}'; registered: {available_backends()}"
        ) from exc


def set_default_backend(name: str) -> None:
    """Set the process-wide default backend (must already be registered)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}'; registered: {available_backends()}")
    _set_global_default(name)


@contextlib.contextmanager
def use_backend(name: str) -> Iterator[Backend]:
    """Temporarily switch the active backend within a ``with`` block (thread-local)."""
    backend = get_backend(name)
    prev = getattr(_state, "override", None)
    _state.override = name
    try:
        yield backend
    finally:
        _state.override = prev


def dispatch(op_name: str, *args: Any, backend: str | None = None, **kwargs: Any) -> Any:
    """Route ``op_name`` to the active (or named) backend and invoke it."""
    return get_backend(backend).get(op_name)(*args, **kwargs)


# --- internal default-tracking helpers ---------------------------------------
#
# The process-wide default is shared across threads (a backend registered on
# the import thread must be visible to workers). Only the `use_backend`
# context-manager override is thread-local, so concurrent threads can target
# different backends without stepping on each other.

_default_lock = threading.Lock()
_default_name: str | None = None


def _global_default() -> str | None:
    return _default_name


def _set_global_default(name: str) -> None:
    global _default_name
    with _default_lock:
        _default_name = name


def _active_name() -> str:
    override = getattr(_state, "override", None)
    if override is not None:
        return override
    default = _global_default()
    if default is None:
        raise RuntimeError("no backend registered; import longhornai.runtime first")
    return default
