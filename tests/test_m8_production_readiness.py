"""M8 exit-gate tests — production inference readiness (PLAN.md §9).

The M8 exit gate is "Production inference readiness criteria met (§9)".
This file pins each readiness criterion as a test plus exercises the
underlying machinery (full perf sweep, persistent autotune cache,
regression check, LonghornInference public API).
"""

import json

import numpy as np
import pytest

import longhornai as lh
from longhornai.benchmarks.full_sweep import (
    PerfRow,
    compare_full_sweep,
    format_full_sweep,
    load_full_sweep_baseline,
    run_full_sweep,
    save_full_sweep_baseline,
)
from longhornai.compiler import (
    TuneCache,
    TuneKey,
    TuningSpace,
    autotune_cached,
    shape_signature,
)
from longhornai.models import LlamaConfig, init_random_weights
from longhornai.runtime import (
    GenerationResult,
    LonghornInference,
    SchedulerConfig,
    available_backends,
    inference_session,
)
from longhornai.validation import (
    format_readiness_card,
    run_readiness_card,
    run_regression_check,
)


# --- Production-inference readiness scorecard (M8 exit gate) ---------------

def test_readiness_card_passes():
    """The aggregated PLAN.md §9 scorecard must be all green."""
    card = run_readiness_card()
    assert card.passed, "\n" + format_readiness_card(card)
    # The scorecard covers at least the documented criteria.
    assert card.num_pass == len(card.criteria)
    assert len(card.criteria) >= 5


def test_regression_check_runs_without_errors():
    """The regression gate must run end-to-end (skipping suites without
    baselines is OK; what matters is no crashes)."""
    report = run_regression_check()
    # No findings expected on a fresh CPU run with empty baselines —
    # the gate just records skipped suites.
    assert isinstance(report.findings, list)
    assert isinstance(report.cross_target_failures, list)


# --- Full performance sweep ------------------------------------------------

def test_full_sweep_runs_and_covers_every_family():
    rows = run_full_sweep(dtype=np.float32)
    families = {r.family for r in rows}
    assert {"gemm", "attention", "paged_attention", "quant", "serving"}.issubset(families)
    assert all(r.latency_ms > 0 for r in rows)


def test_full_sweep_format_renders_table():
    rows = run_full_sweep(dtype=np.float32)
    table = format_full_sweep(rows)
    assert "family" in table
    assert "lat (ms)" in table
    for r in rows:
        assert r.shape in table or any(part in table for part in r.shape.split())


def test_full_sweep_baseline_round_trip(tmp_path, monkeypatch):
    from longhornai.benchmarks import full_sweep
    monkeypatch.setattr(full_sweep, "_BASELINE_PATH",
                         tmp_path / "fs_baseline.json")
    rows = run_full_sweep(dtype=np.float32)
    save_full_sweep_baseline(rows, note="unit-test snapshot")
    loaded = load_full_sweep_baseline()
    assert loaded["schema"] == 1
    assert "unit-test" in loaded["note"]
    assert len(loaded["rows"]) == len(rows)


def test_full_sweep_compare_detects_regression():
    rows = run_full_sweep(dtype=np.float32)
    fake_baseline = {
        "schema": 1,
        "rows": [
            {"family": r.family, "op": r.op, "shape": r.shape,
             "dtype": r.dtype, "latency_ms": r.latency_ms / 10}
            for r in rows
        ],
    }
    cmp = compare_full_sweep(rows, fake_baseline, regression_factor=1.5)
    regressions = [c for c in cmp if c["regressed"]]
    # Every shape should regress vs a 10× faster fake baseline.
    assert len(regressions) == len(rows)


def test_packaged_full_sweep_baseline_is_valid_json():
    from longhornai.benchmarks.full_sweep import baseline_path
    payload = json.loads(baseline_path().read_text())
    assert payload["schema"] == 1
    assert "rows" in payload


# --- Persistent autotune cache --------------------------------------------

def test_tune_cache_round_trip(tmp_path):
    cache_path = tmp_path / "tune_cache.json"
    cache = TuneCache.open(cache_path)
    key = TuneKey(op="gemm", shape=(64, 64, 64),
                   dtype="float32", backend="cpu")
    assert cache.lookup(key) is None
    cache.store(key, {"tile_m": 64, "tile_n": 64}, latency_s=0.001)
    cache.save()

    cache2 = TuneCache.open(cache_path)
    config = cache2.lookup(key)
    assert config == {"tile_m": 64, "tile_n": 64}
    assert len(cache2) == 1
    assert key in cache2.keys()


def test_autotune_cached_caches_winning_config(tmp_path):
    cache = TuneCache.open(tmp_path / "tc.json")
    key = TuneKey(op="gemm", shape=(8, 8, 8),
                   dtype="float32", backend="cpu")
    space = TuningSpace(params={"order": ["a", "b"]})

    a = np.ones((8, 8), dtype=np.float32)
    call_count = [0]

    def factory(order="a"):
        call_count[0] += 1
        return lambda: lh.gemm(a, a)

    config = autotune_cached(
        cache, key, kernel_factory=factory, space=space,
        warmup=0, trials=1,
    )
    assert config in ({"order": "a"}, {"order": "b"})
    first_calls = call_count[0]

    # Second call should hit the cache — factory not invoked.
    config2 = autotune_cached(
        cache, key, kernel_factory=factory, space=space,
        warmup=0, trials=1,
    )
    assert config2 == config
    assert call_count[0] == first_calls


def test_shape_signature_is_a_flat_tuple():
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((8, 16), dtype=np.float32)
    sig = shape_signature(a, b)
    assert sig == (4, 8, 8, 16)


# --- LonghornInference (production runtime API) ----------------------------

def _toy_inference_setup(seed=0):
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )
    weights = init_random_weights(config, dtype=np.float32, seed=seed)
    return config, weights


def test_longhorn_inference_generate_basic():
    config, weights = _toy_inference_setup()
    with LonghornInference(weights, config) as inf:
        out = inf.generate([[1, 2, 3, 4]], max_new_tokens=3)
    assert isinstance(out[0], GenerationResult)
    assert len(out[0].output_ids) == 3
    assert out[0].prompt_length == 4


def test_longhorn_inference_handles_batch_of_prompts():
    config, weights = _toy_inference_setup()
    prompts = [[1, 2, 3, 4], [5, 6, 7], [10, 11]]
    with LonghornInference(weights, config) as inf:
        results = inf.generate(prompts, max_new_tokens=2)
    assert len(results) == 3
    for r, p in zip(results, prompts):
        assert r.prompt_length == len(p)
        assert r.generated_length == 2
        assert all(np.isfinite(t) for t in r.output_ids)


def test_longhorn_inference_session_context_manager():
    config, weights = _toy_inference_setup()
    with inference_session(weights, config) as inf:
        out = inf.generate([[1, 2]], max_new_tokens=2)
    assert len(out[0].output_ids) == 2


def test_longhorn_inference_streaming_yields_tokens():
    config, weights = _toy_inference_setup()
    with LonghornInference(weights, config) as inf:
        tokens = list(inf.stream([1, 2, 3], max_new_tokens=4))
    assert len(tokens) == 4
    assert all(isinstance(t, int) for t in tokens)


@pytest.mark.parametrize("backend", ["cpu", "sim", "rtl", "fpga", "lhsil"])
def test_longhorn_inference_runs_on_every_backend(backend):
    config, weights = _toy_inference_setup()
    with LonghornInference(weights, config) as inf:
        out = inf.generate([[1, 2, 3]], max_new_tokens=2, backend=backend)
    assert len(out[0].output_ids) == 2


def test_longhorn_inference_eos_stops_early():
    config, weights = _toy_inference_setup()
    with LonghornInference(weights, config) as inf:
        # Pick an EOS token equal to the first generated ID; should stop after 1.
        first = inf.generate([[1, 2, 3]], max_new_tokens=5)[0].output_ids[0]
        out = inf.generate(
            [[1, 2, 3]], max_new_tokens=5, eos_token_id=first,
        )
    assert out[0].generated_length == 1
    assert out[0].output_ids[-1] == first


def test_longhorn_inference_outputs_consistent_across_backends():
    """Greedy generation must produce identical token IDs on every backend."""
    config, weights = _toy_inference_setup()
    prompt = [[1, 2, 3, 4]]
    cpu_out = None
    for backend in available_backends():
        with LonghornInference(weights, config) as inf:
            out = inf.generate(prompt, max_new_tokens=3, backend=backend)
        if cpu_out is None:
            cpu_out = out[0].output_ids
        else:
            assert out[0].output_ids == cpu_out, (
                f"{backend} diverged from cpu: {out[0].output_ids} vs {cpu_out}"
            )


# --- five-backend registration ---------------------------------------------

def test_five_pre_silicon_backends_present():
    backends = set(available_backends())
    assert {"cpu", "sim", "rtl", "fpga", "lhsil"}.issubset(backends)
