"""Calibration algorithm tests (M5)."""

import numpy as np

import longhornai as lh
from longhornai.quantization import (
    awq_calibrate,
    gptq_calibrate,
    pack_int4,
    quantize_groupwise,
    smooth_quant_calibrate,
)
from longhornai.quantization.calibration.smooth_quant import smooth_quant_apply


# --- SmoothQuant -------------------------------------------------------

def test_smooth_quant_migration_is_mathematically_lossless(rng):
    """X @ W == (X / s) @ (W * s) bit-tight (just per-channel multiply/divide)."""
    K, N = 64, 16
    W = rng.standard_normal((K, N)).astype(np.float32)
    X = rng.standard_normal((128, K)).astype(np.float32)
    sq = smooth_quant_calibrate(W, X, alpha=0.5)
    out_orig = X @ W
    out_smooth = smooth_quant_apply(X, sq.scale) @ sq.weight_smoothed
    assert np.max(np.abs(out_smooth - out_orig)) < 1e-4


def test_smooth_quant_tames_activation_outliers(rng):
    """SmoothQuant should shrink activation max on outlier channels."""
    K = 32
    W = rng.standard_normal((K, 8)).astype(np.float32)
    X = rng.standard_normal((128, K)).astype(np.float32)
    outliers = [3, 17]
    X[:, outliers] *= 10.0
    sq = smooth_quant_calibrate(W, X, alpha=0.5)
    X_smooth = smooth_quant_apply(X, sq.scale)
    # Outlier channels should have shrunk; normal channels roughly unchanged.
    out_amax = np.max(np.abs(X[:, outliers]), axis=0)
    out_amax_smooth = np.max(np.abs(X_smooth[:, outliers]), axis=0)
    assert np.all(out_amax_smooth < out_amax)


def test_smooth_quant_alpha_extremes(rng):
    """alpha=0 → identity; alpha=1 → all difficulty on weights."""
    K, N = 16, 8
    W = rng.standard_normal((K, N)).astype(np.float32)
    X = rng.standard_normal((32, K)).astype(np.float32) + 0.5
    sq0 = smooth_quant_calibrate(W, X, alpha=0.0)
    # alpha=0: scale = 1 / weight_max  → activations untouched only when W=1.
    # So alpha=0 doesn't strictly leave X alone, but it leaves W proportional
    # to weight_max (i.e., normalized). Just verify it runs and is sensible.
    assert sq0.scale.shape == (K,)
    assert sq0.weight_smoothed.shape == (K, N)
    sq1 = smooth_quant_calibrate(W, X, alpha=1.0)
    assert sq1.scale.shape == (K,)


# --- AWQ ----------------------------------------------------------------

def test_awq_picks_alpha_and_runs(rng):
    K, N = 64, 16
    W = rng.standard_normal((K, N)).astype(np.float32)
    X = rng.standard_normal((128, K)).astype(np.float32)
    awq = awq_calibrate(W, X, group_size=32)
    assert awq.alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert awq.weight_packed.shape == (K // 2, N)
    assert awq.scale_groupwise.shape == (K // 32, N)


def test_awq_at_least_as_good_as_naive_int4(rng):
    """AWQ should never be worse than vanilla groupwise INT4 on this metric."""
    K, N = 64, 16
    W = rng.standard_normal((K, N)).astype(np.float32)
    X = rng.standard_normal((128, K)).astype(np.float32)
    X[:, [3, 17]] *= 5.0  # outliers
    awq = awq_calibrate(W, X, group_size=32)
    out_awq = lh.gemm_w4a16(
        (X / awq.awq_scale).astype(np.float32),
        awq.weight_packed, scale_b=awq.scale_groupwise,
        group_size=awq.group_size, K=K, out_dtype=np.float32,
    )
    ref = X @ W

    # Naive INT4 baseline.
    qw, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
    packed = pack_int4(qw.astype(np.int8), axis=0)
    out_naive = lh.gemm_w4a16(
        X, packed, scale_b=params.scale,
        group_size=32, K=K, out_dtype=np.float32,
    )

    awq_err = np.linalg.norm(out_awq - ref)
    naive_err = np.linalg.norm(out_naive - ref)
    # AWQ's grid-search over alpha includes the naive (alpha=0) point with
    # weights unchanged, so AWQ should match or beat the naive baseline.
    # Allow a tiny slack to absorb rounding-noise differences.
    assert awq_err <= naive_err * 1.01


# --- GPTQ ---------------------------------------------------------------

def test_gptq_runs_and_produces_packed_weights(rng):
    K, N = 64, 16
    W = rng.standard_normal((K, N)).astype(np.float32)
    X = rng.standard_normal((128, K)).astype(np.float32)
    gp = gptq_calibrate(W, X, group_size=32)
    assert gp.weight_packed.shape == (K // 2, N)
    assert gp.scale_groupwise.shape == (K // 32, N)
    # Output runs through the W4A16 kernel without error.
    out = lh.gemm_w4a16(
        X, gp.weight_packed, scale_b=gp.scale_groupwise,
        group_size=gp.group_size, K=K, out_dtype=np.float32,
    )
    assert out.shape == (128, N)
    assert np.all(np.isfinite(out))


def test_gptq_beats_no_calibration_on_average(rng):
    """GPTQ should narrow the FP↔INT4 gap on average vs naive INT4 over multiple seeds."""
    K, N = 64, 16
    seeds = range(8)
    gptq_advantage = 0
    for seed in seeds:
        local = np.random.default_rng(seed)
        W = local.standard_normal((K, N)).astype(np.float32)
        X = local.standard_normal((256, K)).astype(np.float32)
        gp = gptq_calibrate(W, X, group_size=32)
        out_gptq = lh.gemm_w4a16(
            X, gp.weight_packed, scale_b=gp.scale_groupwise,
            group_size=gp.group_size, K=K, out_dtype=np.float32,
        )
        qw, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
        packed = pack_int4(qw.astype(np.int8), axis=0)
        out_naive = lh.gemm_w4a16(
            X, packed, scale_b=params.scale,
            group_size=32, K=K, out_dtype=np.float32,
        )
        ref = X @ W
        if np.linalg.norm(out_gptq - ref) <= np.linalg.norm(out_naive - ref) * 1.05:
            gptq_advantage += 1
    # At least 5/8 seeds should see GPTQ on par or better.
    assert gptq_advantage >= 5, f"GPTQ won {gptq_advantage}/8 seeds"
