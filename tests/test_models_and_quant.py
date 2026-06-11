import numpy as np

import longhornai as lh
from longhornai.models import llama_mlp_block
from longhornai.quantization import quantize_dequantize
from longhornai.compiler import TuningSpace, autotune


def test_llama_mlp_block_shapes_and_finite(rng):
    tokens, hidden, inter = 4, 32, 64
    x = rng.standard_normal((tokens, hidden)).astype(np.float32)
    w_gate = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.1
    w_up = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.1
    w_down = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.1
    out = llama_mlp_block(x, w_gate, w_up, w_down)
    assert out.shape == (tokens, hidden)
    assert np.all(np.isfinite(out))


def test_quantize_dequantize_roundtrip(rng):
    x = rng.standard_normal((64, 64)).astype(np.float32)
    xq = quantize_dequantize(x, bits=8, axis=0)
    # 8-bit symmetric quant should track the original closely in relative terms.
    rel = np.abs(xq - x).max() / np.abs(x).max()
    assert rel < 0.05


def test_autotune_picks_a_config(rng):
    a = rng.standard_normal((128, 128)).astype(np.float32)
    b = rng.standard_normal((128, 128)).astype(np.float32)

    def factory(order="ab"):
        if order == "ab":
            return lambda: lh.gemm(a, b)
        return lambda: lh.gemm(a, b)  # single real path; space exercises the driver

    space = TuningSpace(params={"order": ["ab", "ab2"]})
    result = autotune(factory, space, warmup=1, trials=2)
    assert result.best_config in ({"order": "ab"}, {"order": "ab2"})
    assert result.best_latency_s > 0
