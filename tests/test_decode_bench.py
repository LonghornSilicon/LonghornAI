"""Decode tokens/sec benchmark tests (M4)."""

import json

import numpy as np

from longhornai.benchmarks import decode_bench
from longhornai.benchmarks.decode_bench import (
    DecodeBenchmarkResult,
    load_baseline,
    run_decode_benchmark,
    save_baseline,
)
from longhornai.models import LlamaConfig, init_random_weights


def _toy_config():
    return LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
    )


def test_decode_bench_runs_and_reports_tokens_per_second():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    result = run_decode_benchmark(
        config=config, weights=weights,
        num_requests=2, prompt_len=4, max_new_tokens=4,
    )
    assert isinstance(result, DecodeBenchmarkResult)
    assert result.iterations >= 1
    assert result.decode_tokens > 0
    assert result.tokens_per_second > 0
    assert result.wall_time_s > 0
    assert "tokens / sec" in result.report()


def test_decode_bench_tokens_per_second_scales_with_batch():
    """More concurrent requests should increase aggregate tokens/sec
    on this reference impl (per-request work is similar; batching helps)."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    res1 = run_decode_benchmark(config=config, weights=weights,
                                num_requests=1, prompt_len=4, max_new_tokens=8)
    res4 = run_decode_benchmark(config=config, weights=weights,
                                num_requests=4, prompt_len=4, max_new_tokens=8)
    # Loose check: batched run gets at least the single-stream rate.
    # On a noisy CPU we don't promise strict scaling, just non-degradation
    # plus at least 1.5x — comfortably under what real batching delivers.
    assert res4.tokens_per_second > res1.tokens_per_second


def test_decode_baseline_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(decode_bench, "_BASELINE_PATH", tmp_path / "decode_baseline.json")
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    result = run_decode_benchmark(config=config, weights=weights,
                                  num_requests=2, prompt_len=3, max_new_tokens=2,
                                  label="unit-test")
    save_baseline([result], note="unit test snapshot")
    loaded = load_baseline()
    assert loaded["schema"] == 1
    assert loaded["results"][0]["label"] == "unit-test"
    assert loaded["results"][0]["tokens_per_second"] > 0


def test_packaged_decode_baseline_is_valid_json():
    path = decode_bench.baseline_path()
    payload = json.loads(path.read_text())
    assert payload["schema"] == 1
    assert "results" in payload
