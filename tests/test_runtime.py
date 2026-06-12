import numpy as np
import pytest

import longhornai as lh
from longhornai.runtime import Backend, get_backend, register_backend, use_backend


def test_cpu_backend_registered():
    assert "cpu" in lh.available_backends()
    assert get_backend().name == "cpu"


def test_dispatch_unknown_op_raises():
    with pytest.raises(NotImplementedError):
        lh.dispatch("nonexistent_op", np.zeros(1))


def test_use_backend_context_switch():
    # Register a stub backend that returns a sentinel for gemm.
    stub = Backend("stub", "test backend")
    stub.register("gemm")(lambda a, b, **kw: "STUB")
    register_backend(stub)

    a = np.ones((2, 2), dtype=np.float32)
    assert get_backend().name == "cpu"
    with use_backend("stub"):
        assert lh.dispatch("gemm", a, a) == "STUB"
    # restored after the context
    assert get_backend().name == "cpu"
    assert isinstance(lh.gemm(a, a), np.ndarray)


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        get_backend("does_not_exist")


def test_default_backend_visible_from_worker_thread():
    # The process-wide default must be visible to threads that did not import
    # longhornai themselves; only the use_backend() override is thread-local.
    import threading

    out: list = []
    def worker():
        try:
            out.append(lh.gemm(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)))
        except Exception as e:  # pragma: no cover - failure path
            out.append(e)

    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert isinstance(out[0], np.ndarray), out[0]


def test_use_backend_override_is_thread_local():
    # use_backend in one thread must NOT leak into another thread.
    import threading

    stub = Backend("stub_tl", "thread-local test")
    stub.register("gemm")(lambda a, b, **kw: "STUB_TL")
    register_backend(stub)

    seen: list = []
    started = threading.Event()
    proceed = threading.Event()

    def worker():
        started.set()
        proceed.wait()
        seen.append(get_backend().name)

    t = threading.Thread(target=worker)
    t.start()
    started.wait()
    with use_backend("stub_tl"):
        # main thread sees stub; worker (still parked) must not.
        assert get_backend().name == "stub_tl"
        proceed.set()
        t.join()
    assert seen == ["cpu"]
