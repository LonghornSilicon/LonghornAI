"""RTL backend + cross-target equivalence tests (M3 exit gate).

PLAN.md §8 M3 exit gate: **RTL ≡ CPU on attention**. The cross-target
equivalence harness in `validation/equivalence.py` runs each registered
attention + KV-cache op under both backends and asserts agreement under the
per-dtype tolerance policy (PLAN.md §5.2 keystone test).

Until real Verilator hookups land, the RTL backend is a CPU-equivalent
shim — these tests verify the registration surface and the harness, and
will keep working unchanged when real RTL co-sim arrives.
"""

import numpy as np
import pytest

import longhornai as lh
from longhornai.runtime import get_backend, use_backend
from longhornai.validation import (
    EquivalenceReport,
    assert_cross_target_equivalent,
)


def test_rtl_backend_registered():
    assert "rtl" in lh.available_backends()
    rtl = get_backend("rtl")
    assert rtl.name == "rtl"
    # RTL must implement at least the attention surface required by M3.
    required = {"sdpa", "flash_attention_v1", "flash_attention_v2",
                "multi_head_attention", "kv_cache_append", "kv_cache_gather"}
    assert required.issubset(set(rtl.ops()))


def test_use_backend_switches_to_rtl():
    assert get_backend().name == "cpu"
    with use_backend("rtl"):
        assert get_backend().name == "rtl"
        # ops still work
        a = np.ones((4, 4), dtype=np.float32)
        out = lh.gemm(a, a)
        assert out.shape == (4, 4)
    assert get_backend().name == "cpu"


# --- M3 exit gate: RTL ≡ CPU on every attention op ------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_rtl_eq_cpu_sdpa(rng, dtype, causal):
    q = rng.standard_normal((1, 2, 16, 8)).astype(dtype)
    k = rng.standard_normal((1, 2, 16, 8)).astype(dtype)
    v = rng.standard_normal((1, 2, 16, 8)).astype(dtype)
    report = assert_cross_target_equivalent(
        "sdpa", q, k, v, dtype=dtype, causal=causal,
    )
    assert report.passed, report


@pytest.mark.parametrize("op", ["flash_attention_v1", "flash_attention_v2"])
@pytest.mark.parametrize("causal", [False, True])
def test_rtl_eq_cpu_flash(rng, op, causal):
    fn = getattr(lh, op)  # ensures the kernel name is wired up
    assert callable(fn)
    q = rng.standard_normal((1, 2, 32, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 32, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 32, 8)).astype(np.float32)
    report = assert_cross_target_equivalent(
        op, q, k, v, dtype=np.float32, causal=causal, block_q=8, block_kv=8,
    )
    assert report.passed, report


@pytest.mark.parametrize("attn_impl", ["sdpa", "flash_v1", "flash_v2"])
def test_rtl_eq_cpu_multi_head_attention(rng, attn_impl):
    B, S, Hq, Hkv, D = 1, 16, 4, 2, 8
    q = rng.standard_normal((B, S, Hq * D)).astype(np.float32)
    k = rng.standard_normal((B, S, Hkv * D)).astype(np.float32)
    v = rng.standard_normal((B, S, Hkv * D)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "multi_head_attention", q, k, v, dtype=np.float32,
        num_q_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        causal=True, attn_impl=attn_impl,
    )
    assert report.passed, report


def test_rtl_eq_cpu_kv_cache_append(rng):
    ck = np.zeros((1, 2, 8, 4), dtype=np.float32)
    cv = np.zeros((1, 2, 8, 4), dtype=np.float32)
    k_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    v_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "kv_cache_append", ck, cv, k_new, v_new,
        dtype=np.float32, position=2,
    )
    assert report.passed, report


def test_rtl_eq_cpu_kv_cache_gather(rng):
    ck = rng.standard_normal((1, 2, 8, 4)).astype(np.float32)
    cv = rng.standard_normal((1, 2, 8, 4)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "kv_cache_gather", ck, cv, dtype=np.float32, length=5,
    )
    assert report.passed, report


def test_rtl_eq_cpu_kv_cache_int8_round_trip(rng):
    qk = np.zeros((1, 2, 8, 4), dtype=np.int8)
    sk = np.zeros((1, 2, 8), dtype=np.float32)
    qv = np.zeros((1, 2, 8, 4), dtype=np.int8)
    sv = np.zeros((1, 2, 8), dtype=np.float32)
    k_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    v_new = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "kv_cache_quantize_append",
        qk, sk, qv, sv, k_new, v_new,
        dtype=np.float32, position=0,
    )
    assert report.passed, report


# --- Spot-check that the harness *would* catch a divergence ----------------

def test_equivalence_harness_flags_divergence():
    """Defensive: register a stub backend that returns wrong output and
    confirm the harness flags it. Keeps the harness honest."""
    from longhornai.runtime import Backend, register_backend

    bad = Backend("__bad_for_test__", "intentionally wrong")
    bad.register("gemm")(lambda a, b, **kw: np.zeros_like(a @ b))
    register_backend(bad)

    a = np.ones((4, 4), dtype=np.float32)
    report = assert_cross_target_equivalent(
        "gemm", a, a, dtype=np.float32, backends=["cpu", "__bad_for_test__"],
    )
    assert not report.passed
    bad_outcome = next(o for o in report.outcomes if o.backend == "__bad_for_test__")
    assert not bad_outcome.success
    assert bad_outcome.max_rel_error > 0
