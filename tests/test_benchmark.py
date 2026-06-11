import numpy as np

import longhornai as lh
from longhornai.benchmarks import DevicePeak, benchmark, cost_for
from longhornai.benchmarks.cost import cost_gemm


def test_gemm_cost_model():
    cm = cost_gemm(128, 256, 64, dtype=np.float16)
    assert cm.flops == 2 * 128 * 256 * 64
    # bytes = 2 (fp16) * (M*K + K*N + M*N)
    assert cm.bytes_moved == 2 * (128 * 256 + 256 * 64 + 128 * 64)
    assert cm.arithmetic_intensity > 0


def test_benchmark_produces_metrics(rng):
    m = k = n = 128
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    cost = cost_for("gemm", dtype=np.float16, m=m, k=k, n=n)
    res = benchmark(lambda: lh.gemm(a, b), cost, name="t", warmup=1, trials=3)
    assert res.latency_s > 0
    assert res.achieved_flops > 0
    assert 0 < res.flops_utilization
    assert "roofline" in res.report()


def test_roofline_classification():
    # Small-intensity op is memory bound; large-intensity op compute bound.
    peak = DevicePeak("t", peak_flops=1e12, peak_bandwidth=1e11)  # ridge = 10 FLOP/byte
    big = cost_for("gemm", dtype=np.float16, m=1024, k=1024, n=1024)
    res = benchmark(lambda: None, big, warmup=0, trials=1, peak=peak)
    assert res.is_compute_bound  # 1024^3 GEMM is firmly compute-bound
