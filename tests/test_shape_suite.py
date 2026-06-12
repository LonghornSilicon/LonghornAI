"""Tests for the GEMM shape suite (benchmarks/shape_suite.py)."""

import json

import numpy as np
import pytest

from longhornai.benchmarks import shape_suite as ss
from longhornai.benchmarks.shape_suite import (
    GEMM_SHAPE_SUITE,
    ShapeCase,
    compare_to_baseline,
    format_suite_report,
    load_baseline,
    run_gemm_suite,
    save_baseline,
)


def test_canonical_suite_includes_compute_and_memory_bound_shapes():
    # Compute-bound (square): all dims equal, large.
    assert any(c.M == c.K == c.N >= 1024 for c in GEMM_SHAPE_SUITE)
    # Memory-bound (decode skinny): M == 1.
    assert any(c.M == 1 for c in GEMM_SHAPE_SUITE)
    # No duplicates.
    names = [c.name for c in GEMM_SHAPE_SUITE]
    assert len(set(names)) == len(names)


def test_run_gemm_suite_at_small_scale(rng):
    cases = [ShapeCase("tiny", 16, 16, 16), ShapeCase("rect", 32, 64, 8)]
    results = run_gemm_suite(dtype=np.float32, cases=cases, warmup=1, trials=2)
    assert len(results) == 2
    assert all(r.latency_ms > 0 for r in results)
    assert all(r.achieved_tflops > 0 for r in results)
    # Compute-bound classification is meaningful.
    assert isinstance(results[0].is_compute_bound, bool)


def test_format_suite_report_renders_table():
    cases = [ShapeCase("tiny", 16, 16, 16)]
    results = run_gemm_suite(dtype=np.float32, cases=cases, warmup=0, trials=1)
    report = format_suite_report(results)
    assert "tiny" in report
    assert "lat (ms)" in report
    assert "TFLOP/s" in report


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "_BASELINE_PATH", tmp_path / "baseline.json")
    cases = [ShapeCase("t", 16, 16, 16)]
    res = run_gemm_suite(dtype=np.float32, cases=cases, warmup=0, trials=1)
    save_baseline(res, note="unit-test snapshot")
    loaded = load_baseline()
    assert loaded["schema"] == 1
    assert "unit-test" in loaded["note"]
    assert loaded["shapes"][0]["name"] == "t"


def test_compare_detects_regression():
    cases = [ShapeCase("t", 16, 16, 16)]
    results = run_gemm_suite(dtype=np.float32, cases=cases, warmup=0, trials=1)
    fake_baseline = {
        "schema": 1,
        "shapes": [{"name": "t", "latency_ms": results[0].latency_ms / 10}],
    }
    rows = compare_to_baseline(results, fake_baseline, regression_factor=1.5)
    assert rows[0]["regressed"]
    assert rows[0]["ratio"] > 1.5


def test_compare_passes_when_within_factor():
    cases = [ShapeCase("t", 16, 16, 16)]
    results = run_gemm_suite(dtype=np.float32, cases=cases, warmup=0, trials=1)
    # baseline = current => factor = 1.0; well within 1.5×.
    fake_baseline = {
        "schema": 1,
        "shapes": [{"name": "t", "latency_ms": results[0].latency_ms}],
    }
    rows = compare_to_baseline(results, fake_baseline, regression_factor=1.5)
    assert not rows[0]["regressed"]


def test_compare_treats_unknown_shape_as_clean():
    cases = [ShapeCase("new_shape", 8, 8, 8)]
    results = run_gemm_suite(dtype=np.float32, cases=cases, warmup=0, trials=1)
    rows = compare_to_baseline(results, {"schema": 1, "shapes": []})
    assert not rows[0]["regressed"]
    assert rows[0]["baseline_ms"] is None


def test_load_baseline_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "_BASELINE_PATH", tmp_path / "no_such.json")
    base = load_baseline()
    assert base == {"schema": 1, "shapes": []}


def test_packaged_baseline_is_valid_json():
    # The shipped placeholder must always parse — broken JSON would block CI.
    text = ss.baseline_path().read_text()
    payload = json.loads(text)
    assert payload["schema"] == 1
    assert "shapes" in payload
