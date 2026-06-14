#include <gtest/gtest.h>

#include <cmath>

#include "kernels/dtypes.hpp"
#include "test_util.hpp"

using lh::bfloat16;
using lh::half;

TEST(Dtypes, HalfRoundTrip) {
    // half has ~11 bits of mantissa -> relative error around 2^-11.
    const auto vals = lh_test::random_vector(4096, 1, -100.0f, 100.0f);
    for (float v : vals) {
        const float r = static_cast<float>(half(v));
        EXPECT_LE(std::fabs(r - v), 1e-3f * std::fabs(v) + 1e-4f) << "v=" << v;
    }
}

TEST(Dtypes, Bf16RoundTrip) {
    // bfloat16 keeps 8 mantissa bits -> relative error around 2^-8.
    const auto vals = lh_test::random_vector(4096, 2, -100.0f, 100.0f);
    for (float v : vals) {
        const float r = static_cast<float>(bfloat16(v));
        EXPECT_LE(std::fabs(r - v), 8e-3f * std::fabs(v) + 1e-4f) << "v=" << v;
    }
}

TEST(Dtypes, HalfSpecialValues) {
    EXPECT_EQ(static_cast<float>(half(0.0f)), 0.0f);
    EXPECT_EQ(static_cast<float>(half(1.0f)), 1.0f);
    EXPECT_EQ(static_cast<float>(half(-2.0f)), -2.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(half(70000.0f))));  // overflow
}

TEST(Dtypes, Bf16ExactForFloatPrefix) {
    // Powers of two are representable exactly in bf16.
    EXPECT_EQ(static_cast<float>(bfloat16(0.5f)), 0.5f);
    EXPECT_EQ(static_cast<float>(bfloat16(256.0f)), 256.0f);
    EXPECT_EQ(static_cast<float>(bfloat16(-1.0f)), -1.0f);
}

using lh::fp8_e4m3;
using lh::fp8_e5m2;

TEST(Dtypes, Fp8E4m3RoundTrip) {
    // E4M3 has 3 mantissa bits → relative error around 2^-3.
    const auto vals = lh_test::random_vector(2048, 30, -100.0f, 100.0f);
    int saturated = 0;
    for (float v : vals) {
        const float r = static_cast<float>(fp8_e4m3(v));
        if (std::fabs(r) >= 448.0f) {
            ++saturated;
            continue;
        }
        EXPECT_LE(std::fabs(r - v), 0.13f * std::fabs(v) + 1e-3f) << "v=" << v;
    }
    EXPECT_LT(saturated, 100);  // most values fit comfortably in E4M3 range
}

TEST(Dtypes, Fp8E4m3ExactForRepresentablePowers) {
    EXPECT_EQ(static_cast<float>(fp8_e4m3(0.0f)), 0.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(1.0f)), 1.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(2.0f)), 2.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(-4.0f)), -4.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(0.5f)), 0.5f);
    // Out-of-range saturates to ±max_finite (no infinity in E4M3).
    EXPECT_EQ(static_cast<float>(fp8_e4m3(1000.0f)), 448.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(-1000.0f)), -448.0f);
}

TEST(Dtypes, Fp8E5m2RoundTrip) {
    // E5M2 has 2 mantissa bits → relative error around 2^-2.
    const auto vals = lh_test::random_vector(2048, 31, -1000.0f, 1000.0f);
    for (float v : vals) {
        const float r = static_cast<float>(fp8_e5m2(v));
        if (std::isinf(r)) continue;  // E5M2 has Inf
        EXPECT_LE(std::fabs(r - v), 0.27f * std::fabs(v) + 1e-3f) << "v=" << v;
    }
}

TEST(Dtypes, Fp8E5m2ExactForPowers) {
    EXPECT_EQ(static_cast<float>(fp8_e5m2(1.0f)), 1.0f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2(8.0f)), 8.0f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2(-32.0f)), -32.0f);
    // Overflow -> Inf
    EXPECT_TRUE(std::isinf(static_cast<float>(fp8_e5m2(1.0e6f))));
}
