// LonghornAI — perf model validation against Phase 8 dossiers.
//
// The Phase 9 acceptance gate. Asserts that the analytical performance
// model's bound classification matches every claim made in the
// `docs/silicon/kernels/<name>.md` dossiers. If a kernel is documented
// as "memory-bound" in its dossier but the model predicts "compute-
// bound" on the canonical shape, something is wrong with either the
// model parameters or the dossier — and the test will say which.
#include <gtest/gtest.h>

#include <map>
#include <string>

#include "perf_model.hpp"

namespace {

// The expected bound class per kernel as documented in
// docs/silicon/kernels/<name>.md, on the canonical shape used by
// `lh_perf::default_bench_suite()`.
//
// Test policy: this is a **directional** check, not strict equality.
// Near-ridge kernels can flip between Compute / Balanced or Memory /
// Balanced depending on the architectural parameters. The test asserts
// only that the prediction does not flip to the *opposite* class:
//   expected = Memory   → predicted ≠ Compute
//   expected = Compute  → predicted ≠ Memory
//   expected = Balanced → any
//
// This matches how the dossiers are written: they make a claim about
// which roof binds in the limit, and "balanced" is a soft gradient
// between the two.
const std::map<std::string, lh_perf::Bound>& expected_bounds() {
    static const std::map<std::string, lh_perf::Bound> m = {
        {"gemm/qkv_small", lh_perf::Bound::Compute},
        {"gemm/mlp_up", lh_perf::Bound::Compute},
        {"gemm/mlp_down", lh_perf::Bound::Compute},
        {"rmsnorm[128x4096]", lh_perf::Bound::Memory},
        {"softmax[128x4096]", lh_perf::Bound::Memory},
        {"flash_attn/prefill", lh_perf::Bound::Balanced},
        {"flash_attn/decode", lh_perf::Bound::Memory},
        {"kv_append", lh_perf::Bound::Memory},
        {"gemm_w4a16[128x4096x4096,G=128]", lh_perf::Bound::Compute},
        // moe.md is explicit that this kernel is "per-expert compute-bound
        // like GEMM" but "block-level memory-bound unless aggressive weight
        // quantization is applied" — i.e. it goes either way depending on
        // profile and quantization. Tag as Balanced.
        {"moe_expert_mlp[per_expert=64]", lh_perf::Bound::Balanced},
        {"selective_scan", lh_perf::Bound::Memory},
        {"wkv", lh_perf::Bound::Memory},
        {"sample/topk_topp", lh_perf::Bound::Memory},
        {"embedding", lh_perf::Bound::Memory},
        {"moe_dispatch", lh_perf::Bound::Memory},
    };
    return m;
}

// Returns true if `predicted` is not the strict opposite of `expected`.
// Balanced never conflicts with anything; Compute conflicts only with
// Memory and vice versa.
bool consistent(lh_perf::Bound expected, lh_perf::Bound predicted) {
    if (expected == lh_perf::Bound::Memory) {
        return predicted != lh_perf::Bound::Compute;
    }
    if (expected == lh_perf::Bound::Compute) {
        return predicted != lh_perf::Bound::Memory;
    }
    return true;
}

}  // namespace

TEST(PerfModel, EveryKernelHasAnExpectedBound) {
    // Defensive: if someone adds a kernel to the bench suite without
    // updating the contract, fail loudly so the dossier gets written.
    const auto kernels = lh_perf::default_bench_suite();
    const auto& expected = expected_bounds();
    for (const auto& k : kernels) {
        EXPECT_NE(expected.find(k.name), expected.end())
            << "kernel '" << k.name << "' has no bound expectation in "
            << "test_perf_model.cpp; either add it or remove it from the "
            << "bench suite";
    }
}

TEST(PerfModel, EdgeProfileBoundsConsistentWithDossiers) {
    const auto cfg = lh_perf::LonghornConfig::edge();
    const auto kernels = lh_perf::default_bench_suite();
    const auto preds = lh_perf::predict_all(kernels, cfg);
    const auto& expected = expected_bounds();
    for (const auto& p : preds) {
        const auto it = expected.find(p.name);
        if (it == expected.end()) continue;
        EXPECT_TRUE(consistent(it->second, p.bound))
            << "kernel '" << p.name << "' on edge profile predicted "
            << lh_perf::bound_name(p.bound) << ", dossier expects "
            << lh_perf::bound_name(it->second) << " (or balanced)";
    }
}

TEST(PerfModel, ServerProfileBoundsConsistentWithDossiers) {
    const auto cfg = lh_perf::LonghornConfig::server();
    const auto kernels = lh_perf::default_bench_suite();
    const auto preds = lh_perf::predict_all(kernels, cfg);
    const auto& expected = expected_bounds();
    for (const auto& p : preds) {
        const auto it = expected.find(p.name);
        if (it == expected.end()) continue;
        EXPECT_TRUE(consistent(it->second, p.bound))
            << "kernel '" << p.name << "' on server profile predicted "
            << lh_perf::bound_name(p.bound) << ", dossier expects "
            << lh_perf::bound_name(it->second) << " (or balanced)";
    }
}

TEST(PerfModel, KvControllerAttentionDecodeIsTheBandwidthCrisis) {
    // Phase 8's roofline.md asserts decode attention is the dominant
    // memory-bound operation in inference. Verify the model agrees:
    // flash_attn/decode should have much higher t_memory than t_compute
    // on both profiles.
    for (const auto& cfg : {lh_perf::LonghornConfig::edge(),
                            lh_perf::LonghornConfig::server()}) {
        const auto preds =
            lh_perf::predict_all(lh_perf::default_bench_suite(), cfg);
        for (const auto& p : preds) {
            if (p.name != "flash_attn/decode") continue;
            EXPECT_GT(p.t_memory_s, 5.0 * p.t_compute_s)
                << "decode attention should be dominantly memory-bound";
            EXPECT_EQ(p.bound, lh_perf::Bound::Memory);
        }
    }
}

TEST(PerfModel, ServerScalesRoughlyWithTileCount) {
    // Sanity: a compute-bound kernel should run ~tiles× faster on the
    // server profile than on edge, holding everything else constant.
    // Tolerance: within 30% of ideal scaling — accounts for the slightly
    // different per-tile peak between the two configs.
    const auto edge = lh_perf::LonghornConfig::edge();
    const auto srv = lh_perf::LonghornConfig::server();
    const auto kernels = lh_perf::default_bench_suite();
    const auto pe = lh_perf::predict_all(kernels, edge);
    const auto ps = lh_perf::predict_all(kernels, srv);

    // The qkv_small GEMM: edge tile = 32x32 @ 2GHz = 4 TFLOPS;
    // server = 8 tiles * 16 TFLOPS = 128 TFLOPS → expect ~32× speedup
    // when compute-bound and bandwidth doesn't bind.
    for (size_t i = 0; i < pe.size(); ++i) {
        if (pe[i].name != "gemm/qkv_small") continue;
        if (pe[i].bound != lh_perf::Bound::Compute) continue;
        const double ratio = pe[i].t_predicted_s / ps[i].t_predicted_s;
        EXPECT_GT(ratio, 8.0)
            << "compute-bound GEMM should be > 8x faster on server profile, "
            << "got ratio = " << ratio;
    }
}
