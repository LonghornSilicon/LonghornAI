#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/activation.hpp"
#include "kernels/gemm.hpp"
#include "kernels/moe.hpp"
#include "kernels/softmax.hpp"
#include "test_util.hpp"

namespace {

// Naive per-token MoE reference: route, then for each token loop over its
// top-k experts and apply the SwiGLU MLP independently. The dispatched
// kernel must produce identical output up to fp32 reduction order.
std::vector<float> moe_reference(const std::vector<float>& hidden,
                                 const std::vector<float>& router_w,
                                 const std::vector<float>& W_gate,
                                 const std::vector<float>& W_up,
                                 const std::vector<float>& W_down,
                                 const lh::MoEConfig& cfg,
                                 std::vector<int32_t>* ids_out = nullptr,
                                 std::vector<float>* gates_out = nullptr) {
    const int T = cfg.n_tokens;
    const int H = cfg.hidden_dim;
    const int I = cfg.intermediate_dim;
    const int E = cfg.n_experts;
    const int K = cfg.top_k;

    std::vector<float> logits(T * E, 0.0f);
    lh::gemm(hidden.data(), router_w.data(), logits.data(), T, E, H);

    std::vector<int32_t> ids(T * K);
    std::vector<float> gates(T * K);
    for (int t = 0; t < T; ++t) {
        const float* row = logits.data() + static_cast<int64_t>(t) * E;
        // Top-k by score, lowest-index tie-break.
        std::vector<std::pair<float, int32_t>> ranked(E);
        for (int e = 0; e < E; ++e) ranked[e] = {row[e], e};
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        // Renormalised softmax over the top-k.
        float m = ranked[0].first;
        for (int k = 1; k < K; ++k) m = std::max(m, ranked[k].first);
        float sum = 0.0f;
        std::vector<float> exps(K);
        for (int k = 0; k < K; ++k) {
            exps[k] = std::exp(ranked[k].first - m);
            sum += exps[k];
        }
        const float inv = 1.0f / sum;
        for (int k = 0; k < K; ++k) {
            ids[t * K + k] = ranked[k].second;
            gates[t * K + k] = exps[k] * inv;
        }
    }
    if (ids_out) *ids_out = ids;
    if (gates_out) *gates_out = gates;

    std::vector<float> output(T * H, 0.0f);
    std::vector<float> g_proj(I), u_proj(I), act(I), down(H);
    for (int t = 0; t < T; ++t) {
        const float* x = hidden.data() + static_cast<int64_t>(t) * H;
        for (int k = 0; k < K; ++k) {
            const int32_t e = ids[t * K + k];
            const float gate = gates[t * K + k];
            const float* Wg = W_gate.data() + static_cast<int64_t>(e) * H * I;
            const float* Wu = W_up.data() + static_cast<int64_t>(e) * H * I;
            const float* Wd = W_down.data() + static_cast<int64_t>(e) * I * H;
            lh::gemm(x, Wg, g_proj.data(), 1, I, H);
            lh::gemm(x, Wu, u_proj.data(), 1, I, H);
            for (int i = 0; i < I; ++i) {
                const float s = g_proj[i] / (1.0f + std::exp(-g_proj[i]));
                act[i] = s * u_proj[i];
            }
            lh::gemm(act.data(), Wd, down.data(), 1, H, I);
            float* o = output.data() + static_cast<int64_t>(t) * H;
            for (int i = 0; i < H; ++i) o[i] += gate * down[i];
        }
    }
    return output;
}

}  // namespace

TEST(MoE, RouterTopKMatchesArgsortOnLogits) {
    constexpr int T = 8, H = 16, E = 6, K = 2;
    auto hidden = lh_test::random_vector(T * H, 6000);
    auto rw = lh_test::random_vector(H * E, 6001);

    lh::MoEConfig cfg;
    cfg.n_tokens = T;
    cfg.hidden_dim = H;
    cfg.n_experts = E;
    cfg.top_k = K;

    std::vector<int32_t> ids_kernel(T * K);
    std::vector<float> gates_kernel(T * K);
    lh::moe_router(hidden.data(), rw.data(), ids_kernel.data(),
                   gates_kernel.data(), cfg);

    std::vector<int32_t> ids_ref;
    std::vector<float> gates_ref;
    moe_reference(hidden, rw,
                  /*W_gate=*/std::vector<float>(E * H * 1, 0.0f),
                  /*W_up=*/std::vector<float>(E * H * 1, 0.0f),
                  /*W_down=*/std::vector<float>(E * 1 * H, 0.0f),
                  [&] {
                      lh::MoEConfig c = cfg;
                      c.intermediate_dim = 1;  // dummy; only ids/gates checked
                      return c;
                  }(),
                  &ids_ref, &gates_ref);

    EXPECT_EQ(ids_kernel, ids_ref);
    EXPECT_TRUE(lh_test::AllClose(gates_kernel, gates_ref, 1e-5f, 1e-5f));
    // Gates per token sum to 1.
    for (int t = 0; t < T; ++t) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += gates_kernel[t * K + k];
        EXPECT_NEAR(s, 1.0f, 1e-5f);
    }
}

TEST(MoE, DispatchProducesValidLayoutAndPreservesContent) {
    constexpr int T = 6, H = 4, E = 3, K = 2;
    auto hidden = lh_test::random_vector(T * H, 6100);
    // Hand-built routing: each token's top-k experts.
    std::vector<int32_t> ids = {
        0, 1,
        2, 0,
        1, 2,
        0, 1,
        2, 1,
        0, 2,
    };
    lh::MoEConfig cfg;
    cfg.n_tokens = T;
    cfg.hidden_dim = H;
    cfg.n_experts = E;
    cfg.top_k = K;

    std::vector<int32_t> offsets(E + 1, 0);
    std::vector<int32_t> tok_idx(T * K);
    std::vector<int32_t> sl_idx(T * K);
    std::vector<float> dispatched(T * K * H);
    lh::moe_dispatch(hidden.data(), ids.data(), offsets.data(), tok_idx.data(),
                     sl_idx.data(), dispatched.data(), cfg);

    EXPECT_EQ(offsets[E], T * K);
    // Per-expert counts match the routing table.
    std::vector<int> expected_counts(E, 0);
    for (int s : ids) ++expected_counts[s];
    for (int e = 0; e < E; ++e) {
        EXPECT_EQ(offsets[e + 1] - offsets[e], expected_counts[e]);
    }
    // Each dispatched row equals the corresponding source token's hidden
    // state.
    for (int s = 0; s < T * K; ++s) {
        const int32_t t = tok_idx[s];
        for (int i = 0; i < H; ++i) {
            EXPECT_FLOAT_EQ(dispatched[s * H + i], hidden[t * H + i]);
        }
    }
}

TEST(MoE, ForwardMatchesPerTokenReference) {
    constexpr int T = 5, H = 8, I = 16, E = 4, K = 2;
    auto hidden = lh_test::random_vector(T * H, 6200);
    auto rw = lh_test::random_vector(H * E, 6201);
    auto Wg = lh_test::random_vector(E * H * I, 6202);
    auto Wu = lh_test::random_vector(E * H * I, 6203);
    auto Wd = lh_test::random_vector(E * I * H, 6204);

    lh::MoEConfig cfg;
    cfg.n_tokens = T;
    cfg.hidden_dim = H;
    cfg.intermediate_dim = I;
    cfg.n_experts = E;
    cfg.top_k = K;

    auto ref = moe_reference(hidden, rw, Wg, Wu, Wd, cfg);
    std::vector<float> out(T * H, 0.0f);
    lh::moe_forward(hidden.data(), rw.data(), Wg.data(), Wu.data(), Wd.data(),
                    out.data(), cfg);
    EXPECT_TRUE(lh_test::AllClose(out, ref, 1e-4f, 1e-4f));
}

TEST(MoE, EmptyExpertSkippedCleanly) {
    // Force routing onto a single expert (e=0) and verify the others are
    // skipped without writing into the output.
    constexpr int T = 4, H = 4, I = 8, E = 3, K = 1;
    auto hidden = lh_test::random_vector(T * H, 6300);
    auto Wg = lh_test::random_vector(E * H * I, 6302);
    auto Wu = lh_test::random_vector(E * H * I, 6303);
    auto Wd = lh_test::random_vector(E * I * H, 6304);

    // Router weights skewed so expert 0 wins on every token.
    std::vector<float> rw(H * E, 0.0f);
    for (int h = 0; h < H; ++h) rw[h * E + 0] = 10.0f;

    lh::MoEConfig cfg;
    cfg.n_tokens = T;
    cfg.hidden_dim = H;
    cfg.intermediate_dim = I;
    cfg.n_experts = E;
    cfg.top_k = K;

    std::vector<float> out_kernel(T * H, 0.0f);
    lh::moe_forward(hidden.data(), rw.data(), Wg.data(), Wu.data(), Wd.data(),
                    out_kernel.data(), cfg);
    auto out_ref = moe_reference(hidden, rw, Wg, Wu, Wd, cfg);
    EXPECT_TRUE(lh_test::AllClose(out_kernel, out_ref, 1e-4f, 1e-4f));
}

// Phase 5 acceptance gate: grouped GEMM throughput vs. dense GEMM at the
// same total FLOP count. Under balanced load (each expert sees the same
// number of tokens), the grouped path should sit within a factor of the
// dense path. The PLAN.md target is >= 70%; we measure GFLOPS and assert.
TEST(MoE, GroupedGemmThroughputUnderBalancedLoad) {
    constexpr int E = 4;
    constexpr int per_expert = 32;
    constexpr int M = E * per_expert;
    constexpr int N = 64;
    constexpr int K = 64;

    auto A = lh_test::random_vector(M * K, 6400);
    auto B = lh_test::random_vector(E * K * N, 6401);
    std::vector<float> C_dense(M * N, 0.0f);
    std::vector<float> C_grp(M * N, 0.0f);

    // Dense GEMM that does all the work in one call. We replicate the
    // batched-expert work as one big M = E*per_expert problem multiplying
    // by B[0] (purely a peak-rate proxy; we're measuring throughput).
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 5; ++it) {
        lh::gemm(A.data(), B.data(), C_dense.data(), M, N, K);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double dense_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / 5.0;

    // Grouped GEMM: same total FLOPs split across E expert problems.
    std::vector<const float*> A_ptrs(E);
    std::vector<const float*> B_ptrs(E);
    std::vector<float*> C_ptrs(E);
    std::vector<int> Ms(E), Ns(E), Ks(E);
    for (int e = 0; e < E; ++e) {
        A_ptrs[e] = A.data() + static_cast<int64_t>(e) * per_expert * K;
        B_ptrs[e] = B.data() + static_cast<int64_t>(e) * K * N;
        C_ptrs[e] = C_grp.data() + static_cast<int64_t>(e) * per_expert * N;
        Ms[e] = per_expert;
        Ns[e] = N;
        Ks[e] = K;
    }
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < 5; ++it) {
        lh::gemm_grouped(A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
                         Ms.data(), Ns.data(), Ks.data(), E);
    }
    auto t3 = std::chrono::steady_clock::now();
    const double grouped_us =
        std::chrono::duration<double, std::micro>(t3 - t2).count() / 5.0;

    const double ratio = dense_us / grouped_us;
    EXPECT_GT(ratio, 0.70) << "grouped/dense throughput ratio = " << ratio
                           << " (dense_us=" << dense_us
                           << ", grouped_us=" << grouped_us << ")";
}
