#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "kernels/sampling.hpp"
#include "test_util.hpp"

TEST(Sampling, GreedyReturnsArgmax) {
    std::vector<float> logits = {0.1f, 2.5f, 1.0f, 2.5f, -1.0f};  // tie at 1, 3
    EXPECT_EQ(lh::argmax_sample(logits.data(),
                                static_cast<int>(logits.size())),
              1);  // lowest-index tie wins
}

TEST(Sampling, TopKMasksTail) {
    std::vector<float> logits = {0.1f, 2.5f, 1.0f, 3.0f, -1.0f};
    lh::apply_top_k(logits.data(), 5, /*k=*/2);
    // The top 2 are at indices 3 (3.0) and 1 (2.5); others -> -inf.
    EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0);
    EXPECT_FLOAT_EQ(logits[1], 2.5f);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0);
    EXPECT_FLOAT_EQ(logits[3], 3.0f);
    EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0);
}

TEST(Sampling, TopPKeepsCumulativeMass) {
    // Build a simple distribution where the top 3 cover ~95% of mass.
    std::vector<float> logits = {3.0f, 2.5f, 2.0f, -2.0f, -3.0f};
    auto orig = logits;
    lh::apply_top_p(logits.data(), 5, 0.90f);
    // The fp32 softmax over orig:
    //   p ~ {0.475, 0.288, 0.175, 0.0024, 0.0009} ... cumulative top-3 = 0.94 >= 0.9.
    // So indices 3 and 4 should be -inf, 0/1/2 kept.
    EXPECT_FLOAT_EQ(logits[0], orig[0]);
    EXPECT_FLOAT_EQ(logits[1], orig[1]);
    EXPECT_FLOAT_EQ(logits[2], orig[2]);
    EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0);
    EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0);
}

TEST(Sampling, MinPRelativeToMax) {
    std::vector<float> logits = {3.0f, 2.0f, 0.0f, -2.0f};
    lh::apply_min_p(logits.data(), 4, 0.10f);
    // max prob is at logit 3.0; threshold is 0.1 of that probability,
    // i.e. logit - 3.0 >= log(0.1) ~ -2.30.
    // Logits 3.0, 2.0 pass (deltas 0, -1). Logit 0.0 (delta -3) fails.
    // Logit -2.0 (delta -5) fails.
    EXPECT_FLOAT_EQ(logits[0], 3.0f);
    EXPECT_FLOAT_EQ(logits[1], 2.0f);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0);
    EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0);
}

TEST(Sampling, TemperatureZeroIsGreedy) {
    std::vector<float> logits = {0.1f, 2.5f, 1.0f, 3.0f};
    std::vector<float> scratch(4);
    lh::SamplingPolicy p;
    p.temperature = 0.0f;
    uint64_t rng = 12345;
    EXPECT_EQ(lh::sample(logits.data(), 4, p, &rng, scratch.data()), 3);
}

TEST(Sampling, MultinomialConvergesToTrueDistribution) {
    // Two-token distribution with known prob p_true. Sample N times,
    // assert empirical mean within tolerance. Variance ~ p(1-p)/N → for
    // p=0.7, N=20000, std ≈ 0.0032; we use ±0.02 to be safe under
    // splitmix64's distribution.
    std::vector<float> logits(2);
    const float p_true = 0.7f;
    logits[0] = std::log(p_true);
    logits[1] = std::log(1.0f - p_true);
    uint64_t rng = 0xfeedfaceULL;
    constexpr int N = 20000;
    int hits = 0;
    for (int i = 0; i < N; ++i) {
        if (lh::softmax_sample(logits.data(), 2, &rng) == 0) ++hits;
    }
    const double empirical = static_cast<double>(hits) / N;
    EXPECT_NEAR(empirical, static_cast<double>(p_true), 0.02);
}

TEST(Sampling, FullPolicyDeterministicWithSameSeed) {
    auto logits = lh_test::random_vector(64, 7000);
    std::vector<float> s1(64), s2(64);
    lh::SamplingPolicy p;
    p.temperature = 0.8f;
    p.top_k = 16;
    p.top_p = 0.95f;
    uint64_t r1 = 99, r2 = 99;
    const int t1 = lh::sample(logits.data(), 64, p, &r1, s1.data());
    const int t2 = lh::sample(logits.data(), 64, p, &r2, s2.data());
    EXPECT_EQ(t1, t2);
}

TEST(Sampling, SplitmixUniformRange) {
    uint64_t r = 42;
    for (int i = 0; i < 1000; ++i) {
        const double u = lh::uniform01(&r);
        EXPECT_GE(u, 0.0);
        EXPECT_LT(u, 1.0);
    }
}
