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
