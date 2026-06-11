import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# Dtypes the CPU backend supports natively in M1.
DTYPES = [np.float32, np.float16, np.float64]
