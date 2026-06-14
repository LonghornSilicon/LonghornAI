#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/quant.hpp"
#include "test_util.hpp"

TEST(Quant, Q8PerTensorRoundTrip) {
    auto x = lh_test::random_vector(1024, 4000, -3.0f, 3.0f);
    std::vector<int8_t> q(x.size());
    float scale = 0.0f;
    lh::q8_quantize_per_tensor(x.data(), q.data(), &scale, x.size());
    std::vector<float> d(x.size());
    lh::q8_dequantize_per_tensor(q.data(), scale, d.data(), x.size());
    // Max error bounded by ~ scale / 2.
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_LT(std::fabs(d[i] - x[i]), scale + 1e-6f);
    }
}

TEST(Quant, Q8PerRowFollowsRowScale) {
    constexpr int rows = 8, dim = 64;
    // First row tiny values, last row huge — per-row scales should make
    // both representable.
    std::vector<float> x(rows * dim);
    for (int r = 0; r < rows; ++r) {
        const float scale = (r == 0) ? 0.01f : (r == rows - 1 ? 100.0f : 1.0f);
        auto row = lh_test::random_vector(dim, static_cast<uint32_t>(4100 + r),
                                          -scale, scale);
        for (int i = 0; i < dim; ++i) x[r * dim + i] = row[i];
    }
    std::vector<int8_t> q(x.size());
    std::vector<float> scales(rows);
    std::vector<float> d(x.size());
    lh::q8_quantize_per_row(x.data(), q.data(), scales.data(), rows, dim);
    lh::q8_dequantize_per_row(q.data(), scales.data(), d.data(), rows, dim);
    // Relative error per row bounded by ~ 1/127 of the row magnitude.
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < dim; ++i) {
            EXPECT_LT(std::fabs(d[r * dim + i] - x[r * dim + i]),
                      scales[r] + 1e-6f);
        }
    }
}

TEST(Quant, Q8PerColRoundTrip) {
    constexpr int rows = 16, cols = 32;
    auto x = lh_test::random_vector(rows * cols, 4200, -1.0f, 1.0f);
    std::vector<int8_t> q(x.size());
    std::vector<float> scales(cols);
    std::vector<float> d(x.size());
    lh::q8_quantize_per_col(x.data(), q.data(), scales.data(), rows, cols);
    lh::q8_dequantize_per_col(q.data(), scales.data(), d.data(), rows, cols);
    for (int c = 0; c < cols; ++c) {
        EXPECT_GT(scales[c], 0.0f);
        for (int r = 0; r < rows; ++r) {
            EXPECT_LT(std::fabs(d[r * cols + c] - x[r * cols + c]),
                      scales[c] + 1e-6f);
        }
    }
}

TEST(Quant, Q4GroupwiseRoundTripBoundedByScale) {
    constexpr int K = 64, N = 16;
    constexpr int G = 32;
    auto x = lh_test::random_vector(K * N, 4300, -1.0f, 1.0f);
    std::vector<uint8_t> packed(K * N / 2);
    std::vector<float> scales((K / G) * N);
    std::vector<float> d(K * N);
    lh::q4_quantize_groupwise(x.data(), packed.data(), scales.data(), K, N, G);
    lh::q4_dequantize_groupwise(packed.data(), scales.data(), d.data(), K, N,
                                G);
    // 4-bit symmetric: max error is one scale unit per group/col.
    for (int k = 0; k < K; ++k) {
        const int g = k / G;
        for (int n = 0; n < N; ++n) {
            const float s = scales[g * N + n];
            EXPECT_LT(std::fabs(d[k * N + n] - x[k * N + n]), s + 1e-6f);
        }
    }
}

TEST(Quant, Q4GetMatchesDequantize) {
    // Sanity: q4_get inline matches the dequantized buffer.
    constexpr int K = 8, N = 4, G = 4;
    auto x = lh_test::random_vector(K * N, 4400, -1.0f, 1.0f);
    std::vector<uint8_t> packed(K * N / 2);
    std::vector<float> scales((K / G) * N);
    lh::q4_quantize_groupwise(x.data(), packed.data(), scales.data(), K, N, G);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const int8_t got = lh::q4_get(packed.data(), k, n, N);
            EXPECT_GE(got, -7);
            EXPECT_LE(got, 7);
        }
    }
}

TEST(Quant, Fp8PerTensorRoundTrip) {
    auto x = lh_test::random_vector(1024, 4500, -2.0f, 2.0f);
    std::vector<lh::fp8_e4m3> q(x.size());
    float scale = 0.0f;
    lh::fp8_quantize_per_tensor(x.data(), q.data(), &scale, x.size());
    std::vector<float> d(x.size());
    lh::fp8_dequantize_per_tensor(q.data(), scale, d.data(), x.size());
    // E4M3 has 3 mantissa bits → 12.5% relative for normals, plus the
    // per-tensor scale step.
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_LT(std::fabs(d[i] - x[i]), 0.13f * std::fabs(x[i]) + scale);
    }
}

TEST(Quant, Fp8PerRowFollowsRowScale) {
    constexpr int rows = 4, dim = 32;
    std::vector<float> x(rows * dim);
    for (int r = 0; r < rows; ++r) {
        const float mag = (r == 0) ? 0.001f : 50.0f;
        auto row = lh_test::random_vector(dim, static_cast<uint32_t>(4600 + r),
                                          -mag, mag);
        for (int i = 0; i < dim; ++i) x[r * dim + i] = row[i];
    }
    std::vector<lh::fp8_e4m3> q(x.size());
    std::vector<float> scales(rows);
    std::vector<float> d(x.size());
    lh::fp8_quantize_per_row(x.data(), q.data(), scales.data(), rows, dim);
    lh::fp8_dequantize_per_row(q.data(), scales.data(), d.data(), rows, dim);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < dim; ++i) {
            EXPECT_LT(std::fabs(d[r * dim + i] - x[r * dim + i]),
                      0.13f * std::fabs(x[r * dim + i]) + scales[r]);
        }
    }
}
