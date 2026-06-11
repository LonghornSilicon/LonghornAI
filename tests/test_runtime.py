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
