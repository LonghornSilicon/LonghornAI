#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/conv1d.hpp"
#include "kernels/linear_attn.hpp"
#include "test_util.hpp"

TEST(LinearAttention, MatchesPerStepRecurrenceReference) {
    // The reference is the same recurrence formulated explicitly here.
    // This catches indexing/strides bugs by comparing against a copy of
    // the math written in a different style (no scratch reuse).
    constexpr int B = 2, H = 2, L = 5, Df = 4, Dv = 6;
    auto Q = lh_test::random_vector(B * H * L * Df, 9400);
    auto K = lh_test::random_vector(B * H * L * Df, 9401);
    auto V = lh_test::random_vector(B * H * L * Dv, 9402);

    lh::LinearAttnConfig cfg;
    cfg.batch = B;
    cfg.n_heads = H;
    cfg.seq = L;
    cfg.d_feat = Df;
    cfg.d_v = Dv;
    cfg.normalize = true;

    std::vector<float> y(B * H * L * Dv, 0.0f);
    lh::linear_attention(Q.data(), K.data(), V.data(), y.data(), cfg);

    // Reference: one rank-1 update per step, recompute output.
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::vector<float> S(Df * Dv, 0.0f);
            std::vector<float> z(Df, 0.0f);
            for (int t = 0; t < L; ++t) {
                const float* qt = Q.data() +
                    (((b * H + h) * L) + t) * Df;
                const float* kt = K.data() +
                    (((b * H + h) * L) + t) * Df;
                const float* vt = V.data() +
                    (((b * H + h) * L) + t) * Dv;
                for (int f = 0; f < Df; ++f) {
                    z[f] += kt[f];
                    for (int d = 0; d < Dv; ++d) S[f * Dv + d] += kt[f] * vt[d];
                }
                std::vector<float> y_ref(Dv, 0.0f);
                float zsum = 0.0f;
                for (int f = 0; f < Df; ++f) {
                    zsum += qt[f] * z[f];
                    for (int d = 0; d < Dv; ++d)
                        y_ref[d] += qt[f] * S[f * Dv + d];
                }
                const float inv = (zsum > 0.0f) ? (1.0f / zsum) : 0.0f;
                for (int d = 0; d < Dv; ++d) {
                    const float got =
                        y[((b * H + h) * L + t) * Dv + d];
                    EXPECT_NEAR(got, y_ref[d] * inv, 1e-4f);
                }
            }
        }
    }
}

TEST(LinearAttention, NoNormalizeProducesRawNumerator) {
    // With normalize=false the kernel should return q · S (no division).
    constexpr int B = 1, H = 1, L = 3, Df = 2, Dv = 2;
    std::vector<float> Q = {1, 0,
                            0, 1,
                            1, 1};
    std::vector<float> K = {1, 0,
                            0, 1,
                            1, 1};
    std::vector<float> V = {1, 0,
                            0, 1,
                            1, 1};

    lh::LinearAttnConfig cfg;
    cfg.batch = B;
    cfg.n_heads = H;
    cfg.seq = L;
    cfg.d_feat = Df;
    cfg.d_v = Dv;
    cfg.normalize = false;

    std::vector<float> y(L * Dv, 0.0f);
    lh::linear_attention(Q.data(), K.data(), V.data(), y.data(), cfg);

    // After step 0: S = (1,0)⊗(1,0) = [[1,0],[0,0]]; q0=(1,0); y0 = (1,0).
    EXPECT_NEAR(y[0], 1.0f, 1e-6f);
    EXPECT_NEAR(y[1], 0.0f, 1e-6f);
    // Step 1: S += (0,1)⊗(0,1); S = [[1,0],[0,1]]; q1=(0,1); y1 = (0,1).
    EXPECT_NEAR(y[2], 0.0f, 1e-6f);
    EXPECT_NEAR(y[3], 1.0f, 1e-6f);
    // Step 2: S += (1,1)⊗(1,1); S = [[2,1],[1,2]]; q2=(1,1); y2 = (3, 3).
    EXPECT_NEAR(y[4], 3.0f, 1e-6f);
    EXPECT_NEAR(y[5], 3.0f, 1e-6f);
}

TEST(Conv1d, CausalDepthwiseAgreesWithFormula) {
    constexpr int B = 1, L = 6, C = 2, K = 3;
    auto x = lh_test::random_vector(B * L * C, 9500);
    auto w = lh_test::random_vector(C * K, 9501);
    auto bias = lh_test::random_vector(C, 9502);

    lh::Conv1dConfig cfg;
    cfg.batch = B;
    cfg.seq = L;
    cfg.channels = C;
    cfg.kernel_size = K;

    std::vector<float> y(B * L * C, 0.0f);
    lh::conv1d_causal_depthwise(x.data(), w.data(), bias.data(), y.data(),
                                cfg);

    // Brute-force reference, with explicit causal padding.
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float expected = bias[c];
            for (int k = 0; k < K; ++k) {
                const int s = t - k;
                if (s < 0) continue;
                expected += x[s * C + c] * w[c * K + k];
            }
            EXPECT_NEAR(y[t * C + c], expected, 1e-6f);
        }
    }
}

TEST(Conv1d, FirstStepUsesOnlyFirstWeight) {
    // At t=0 the only valid input is x[0]; everything else is zero-padded.
    constexpr int B = 1, L = 1, C = 2, K = 4;
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> w = {0.5f, 9.9f, 9.9f, 9.9f,    // c=0
                            0.25f, 9.9f, 9.9f, 9.9f};  // c=1
    lh::Conv1dConfig cfg;
    cfg.batch = B;
    cfg.seq = L;
    cfg.channels = C;
    cfg.kernel_size = K;
    std::vector<float> y(B * L * C, 0.0f);
    lh::conv1d_causal_depthwise(x.data(), w.data(), nullptr, y.data(), cfg);
    EXPECT_FLOAT_EQ(y[0], 0.5f * 1.0f);
    EXPECT_FLOAT_EQ(y[1], 0.25f * 2.0f);
}
