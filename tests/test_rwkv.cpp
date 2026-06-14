#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/rwkv.hpp"
#include "test_util.hpp"

namespace {

// Naive WKV reference: at each step compute the full numerator and
// denominator over the past window. O(L^2) per channel; only used to
// validate the recurrent kernel for short sequences.
void wkv_naive(const std::vector<float>& k, const std::vector<float>& v,
               const std::vector<float>& w, const std::vector<float>& u,
               std::vector<float>& y, const lh::WkvConfig& cfg) {
    const int L = cfg.seq;
    const int C = cfg.n_channels;
    for (int b = 0; b < cfg.batch; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) {
                // Find max exponent for stability.
                float m = u[c] + k[(b * L + t) * C + c];
                for (int i = 0; i < t; ++i) {
                    const float e = -static_cast<float>(t - i) * w[c] +
                                    k[(b * L + i) * C + c];
                    if (e > m) m = e;
                }
                float num = 0.0f, den = 0.0f;
                for (int i = 0; i < t; ++i) {
                    const float e = -static_cast<float>(t - i) * w[c] +
                                    k[(b * L + i) * C + c];
                    const float p = std::exp(e - m);
                    num += p * v[(b * L + i) * C + c];
                    den += p;
                }
                const float pp = std::exp(u[c] + k[(b * L + t) * C + c] - m);
                num += pp * v[(b * L + t) * C + c];
                den += pp;
                y[(b * L + t) * C + c] = (den > 0.0f) ? (num / den) : 0.0f;
            }
        }
    }
}

}  // namespace

TEST(RWKV, RecurrentMatchesNaiveOnShortSequences) {
    constexpr int B = 1, L = 16, C = 4;
    auto kk = lh_test::random_vector(B * L * C, 9300, -2.0f, 2.0f);
    auto vv = lh_test::random_vector(B * L * C, 9301, -1.0f, 1.0f);
    auto ww = lh_test::random_vector(C, 9302, 0.05f, 0.5f);  // positive decay
    auto uu = lh_test::random_vector(C, 9303, -0.5f, 0.5f);

    lh::WkvConfig cfg;
    cfg.batch = B;
    cfg.seq = L;
    cfg.n_channels = C;

    std::vector<float> y_naive(B * L * C);
    std::vector<float> y_rec(B * L * C);
    wkv_naive(kk, vv, ww, uu, y_naive, cfg);
    lh::wkv(kk.data(), vv.data(), ww.data(), uu.data(), y_rec.data(), cfg);
    EXPECT_TRUE(lh_test::AllClose(y_rec, y_naive, 1e-4f, 1e-4f));
}

TEST(RWKV, StableOnLongSequencesWithLargeKeys) {
    // Keys are huge; without log-space stability the exponent overflows
    // around L=200. Verify the recurrent kernel doesn't overflow at L=2k.
    constexpr int B = 1, L = 2048, C = 2;
    std::vector<float> kk(B * L * C);
    std::vector<float> vv(B * L * C);
    std::vector<float> ww = {0.1f, 0.2f};
    std::vector<float> uu = {0.1f, -0.1f};
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            // Slow positive drift in keys to stress the log-max bookkeeping.
            kk[(t)*C + c] = 50.0f + 0.01f * t + 0.1f * c;
            vv[(t)*C + c] = (t % 7) * 0.1f - (c % 3) * 0.05f;
        }
    }
    lh::WkvConfig cfg;
    cfg.batch = B;
    cfg.seq = L;
    cfg.n_channels = C;
    std::vector<float> y(B * L * C);
    lh::wkv(kk.data(), vv.data(), ww.data(), uu.data(), y.data(), cfg);
    // None of the outputs should be NaN or Inf.
    for (float yy : y) {
        EXPECT_TRUE(std::isfinite(yy));
    }
}
