#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/rope.hpp"
#include "test_util.hpp"

TEST(Rope, OptimizedMatchesReferenceInterleaved) {
    const int seq = 12, n_heads = 4, head_dim = 64;
    auto base = lh_test::random_vector(seq * n_heads * head_dim, 501);
    auto x = base, xref = base;
    lh::rope(x.data(), seq, n_heads, head_dim, 10000.0f, 1.0f, true, 0);
    lh::rope_ref(xref.data(), seq, n_heads, head_dim, 10000.0f, 1.0f, true, 0);
    EXPECT_TRUE(lh_test::AllClose(x, xref));
}

TEST(Rope, OptimizedMatchesReferenceHalfRotation) {
    const int seq = 9, n_heads = 3, head_dim = 48;
    auto base = lh_test::random_vector(seq * n_heads * head_dim, 502);
    auto x = base, xref = base;
    lh::rope(x.data(), seq, n_heads, head_dim, 10000.0f, 0.5f, false, 2);
    lh::rope_ref(xref.data(), seq, n_heads, head_dim, 10000.0f, 0.5f, false, 2);
    EXPECT_TRUE(lh_test::AllClose(x, xref));
}

TEST(Rope, PreservesNorm) {
    // Rotation is orthogonal, so each rotated pair preserves its 2-norm.
    const int seq = 5, n_heads = 2, head_dim = 32;
    auto base = lh_test::random_vector(seq * n_heads * head_dim, 503);
    auto x = base;
    lh::rope(x.data(), seq, n_heads, head_dim);
    for (size_t i = 0; i < base.size(); i += 2) {
        const double n0 = base[i] * base[i] + base[i + 1] * base[i + 1];
        const double n1 = x[i] * x[i] + x[i + 1] * x[i + 1];
        EXPECT_NEAR(n0, n1, 1e-3);
    }
}
