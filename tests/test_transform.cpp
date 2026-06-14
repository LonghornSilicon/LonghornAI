#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "kernels/transform.hpp"
#include "test_util.hpp"

TEST(Transform, Transpose2dRoundTrip) {
    constexpr int M = 5, N = 7;
    auto x = lh_test::random_vector(M * N, 100);
    std::vector<float> y(N * M);
    std::vector<float> z(M * N);
    lh::transpose2d(x.data(), y.data(), M, N);
    lh::transpose2d(y.data(), z.data(), N, M);
    EXPECT_TRUE(lh_test::AllClose(z, x));
}

TEST(Transform, PermuteIdentity) {
    constexpr int A = 3, B = 4, C = 2;
    auto x = lh_test::random_vector(A * B * C, 200);
    std::vector<float> y(A * B * C);
    int64_t shape[3] = {A, B, C};
    int perm[3] = {0, 1, 2};
    lh::permute(x.data(), y.data(), shape, perm, 3);
    EXPECT_TRUE(lh_test::AllClose(y, x));
}

TEST(Transform, PermuteSwapsAxes) {
    constexpr int A = 2, B = 3, C = 4;
    std::vector<float> x(A * B * C);
    for (int i = 0; i < A * B * C; ++i) x[i] = static_cast<float>(i);
    std::vector<float> y(A * B * C);
    int64_t shape[3] = {A, B, C};
    int perm[3] = {0, 2, 1};  // [A, B, C] -> [A, C, B]
    lh::permute(x.data(), y.data(), shape, perm, 3);

    // Spot-check: y[a, c, b] == x[a, b, c]
    for (int a = 0; a < A; ++a)
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c) {
                EXPECT_FLOAT_EQ(y[(a * C + c) * B + b],
                                x[(a * B + b) * C + c]);
            }
}

TEST(Transform, ConcatSplitRoundTrip) {
    constexpr int rank = 3;
    int64_t s1[rank] = {2, 3, 4};
    int64_t s2[rank] = {2, 5, 4};
    int64_t s3[rank] = {2, 2, 4};
    int64_t out_shape[rank] = {2, 10, 4};
    auto x1 = lh_test::random_vector(2 * 3 * 4, 1);
    auto x2 = lh_test::random_vector(2 * 5 * 4, 2);
    auto x3 = lh_test::random_vector(2 * 2 * 4, 3);

    std::vector<float> y(2 * 10 * 4);
    const float* xs[3] = {x1.data(), x2.data(), x3.data()};
    const int64_t* ss[3] = {s1, s2, s3};
    lh::concat(xs, ss, y.data(), out_shape, rank, 3, /*axis=*/1);

    std::vector<float> b1(2 * 3 * 4), b2(2 * 5 * 4), b3(2 * 2 * 4);
    float* ys[3] = {b1.data(), b2.data(), b3.data()};
    int64_t sizes[3] = {3, 5, 2};
    lh::split(y.data(), out_shape, ys, sizes, rank, 3, /*axis=*/1);

    EXPECT_TRUE(lh_test::AllClose(b1, x1));
    EXPECT_TRUE(lh_test::AllClose(b2, x2));
    EXPECT_TRUE(lh_test::AllClose(b3, x3));
}

TEST(Transform, GatherRowsMatchesEmbedding) {
    constexpr int vocab = 8, dim = 5, n = 4;
    auto table = lh_test::random_vector(vocab * dim, 50);
    std::vector<int32_t> ids = {2, 0, 7, 4};
    std::vector<float> y(n * dim);
    lh::gather_rows(table.data(), ids.data(), y.data(), n, vocab, dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            EXPECT_FLOAT_EQ(y[i * dim + d], table[ids[i] * dim + d]);
        }
    }
}

TEST(Transform, GatherRowsOutOfRangeProducesZero) {
    constexpr int vocab = 4, dim = 3, n = 2;
    auto table = lh_test::random_vector(vocab * dim, 51);
    std::vector<int32_t> ids = {-1, 99};
    std::vector<float> y(n * dim);
    lh::gather_rows(table.data(), ids.data(), y.data(), n, vocab, dim);
    for (int i = 0; i < n * dim; ++i) EXPECT_FLOAT_EQ(y[i], 0.0f);
}

TEST(Transform, ScatterAccumulatesDuplicates) {
    constexpr int vocab = 3, dim = 2, n = 4;
    std::vector<float> x = {1.0f, 2.0f,
                            3.0f, 4.0f,
                            5.0f, 6.0f,
                            7.0f, 8.0f};
    std::vector<int32_t> ids = {1, 0, 1, 2};
    std::vector<float> y(vocab * dim, 0.0f);
    lh::scatter_add_rows(x.data(), ids.data(), y.data(), n, vocab, dim);
    EXPECT_FLOAT_EQ(y[0], 3.0f);  // x[1]
    EXPECT_FLOAT_EQ(y[1], 4.0f);
    EXPECT_FLOAT_EQ(y[2], 1.0f + 5.0f);  // x[0] + x[2]
    EXPECT_FLOAT_EQ(y[3], 2.0f + 6.0f);
    EXPECT_FLOAT_EQ(y[4], 7.0f);  // x[3]
    EXPECT_FLOAT_EQ(y[5], 8.0f);
}
