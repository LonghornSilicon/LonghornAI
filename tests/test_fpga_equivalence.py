"""FPGA backend + cross-target equivalence tests (M4 exit gate component).

PLAN.md §8 M4 deliverable: "first FPGA bring-up"; until the FPGA bitstream
flow is wired up the backend is a CPU-equivalent shim, exercised by the
cross-target equivalence harness alongside CPU / sim / RTL.
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.runtime import get_backend, use_backend
from longhornai.validation import assert_cross_target_equivalent


def test_fpga_and_sim_backends_registered():
    backends = lh.available_backends()
    assert "fpga" in backends
    assert "sim" in backends
    fpga = get_backend("fpga")
    sim = get_backend("sim")
    # Each pre-silicon shim implements the full op surface.
    cpu_ops = set(get_backend("cpu").ops())
    assert set(fpga.ops()) == cpu_ops
    assert set(sim.ops()) == cpu_ops


def test_use_backend_switches_to_fpga():
    assert get_backend().name == "cpu"
    with use_backend("fpga"):
        assert get_backend().name == "fpga"
        a = np.ones((4, 4), dtype=np.float32)
        out = lh.gemm(a, a)
        assert out.shape == (4, 4)
    assert get_backend().name == "cpu"


@pytest.mark.parametrize("op", ["sdpa", "flash_attention_v2"])
def test_fpga_eq_cpu_attention(rng, op):
    fn = getattr(lh, op)
    assert callable(fn)
    q = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    report = assert_cross_target_equivalent(
        op, q, k, v, dtype=np.float32, causal=True,
        backends=["cpu", "fpga"],
    )
    assert report.passed, report


def test_fpga_eq_cpu_paged_attention(rng):
    """Paged attention must agree on FPGA — the M4 inner loop runs through it."""
    B, S_q, H_q, H_kv, D = 1, 4, 4, 2, 8
    block_size = 4
    q = rng.standard_normal((B, S_q, H_q * D)).astype(np.float32)
    ck = rng.standard_normal((4, H_kv, block_size, D)).astype(np.float32)
    cv = rng.standard_normal((4, H_kv, block_size, D)).astype(np.float32)
    block_table = np.array([[0, 1, -1]], dtype=np.int32)
    seq_lens = np.array([S_q], dtype=np.int32)
    report = assert_cross_target_equivalent(
        "paged_attention", q, ck, cv,
        dtype=np.float32,
        block_table=block_table, seq_lens=seq_lens,
        block_size=block_size,
        num_q_heads=H_q, num_kv_heads=H_kv, head_dim=D, causal=True,
        backends=["cpu", "fpga"],
    )
    assert report.passed, report


def test_fpga_eq_cpu_paged_kv_append(rng):
    block_size = 4
    ck = np.zeros((4, 2, block_size, 8), dtype=np.float32)
    cv = np.zeros((4, 2, block_size, 8), dtype=np.float32)
    block_table = np.array([[0, 1, -1]], dtype=np.int32)
    seq_lens = np.zeros(1, dtype=np.int32)
    k_new = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)
    v_new = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "paged_kv_append", ck, cv, k_new, v_new,
        dtype=np.float32,
        block_table=block_table, seq_lens=seq_lens, block_size=block_size,
        backends=["cpu", "fpga"],
    )
    assert report.passed, report


def test_all_four_pre_silicon_backends_agree(rng):
    """The full pre-silicon set (cpu / sim / rtl / fpga) must agree on attention."""
    q = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "flash_attention_v2", q, k, v, dtype=np.float32, causal=True,
        backends=["cpu", "sim", "rtl", "fpga"],
    )
    assert report.passed, report
    # All four backends present in the report.
    seen = {o.backend for o in report.outcomes}
    assert {"cpu", "sim", "rtl", "fpga"}.issubset(seen)
