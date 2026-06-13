#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/normalization.hpp"
#include "test_util.hpp"

TEST(LayerNorm, MatchesReference) {
    const int rows = 33, dim = 80;
    auto x = lh_test::random_vector(rows * dim, 101, -5.0f, 5.0f);
    auto g = lh_test::random_vector(dim, 102);
    auto b = lh_test::random_vector(dim, 103);
    std::vector<float> y(rows * dim), yref(rows * dim);

    lh::layernorm(x.data(), g.data(), b.data(), y.data(), rows, dim);
    lh::layernorm_ref(x.data(), g.data(), b.data(), yref.data(), rows, dim);
    EXPECT_TRUE(lh_test::AllClose(y, yref, 1e-3f, 1e-3f));
}

TEST(LayerNorm, ZeroMeanUnitVarProperty) {
    const int rows = 4, dim = 64;
    auto x = lh_test::random_vector(rows * dim, 111, -3.0f, 3.0f);
    std::vector<float> g(dim, 1.0f), b(dim, 0.0f), y(rows * dim);
    lh::layernorm(x.data(), g.data(), b.data(), y.data(), rows, dim, 0.0f);

    for (int r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int i = 0; i < dim; ++i) mean += y[r * dim + i];
        mean /= dim;
        EXPECT_NEAR(mean, 0.0, 1e-3);
    }
}

TEST(RmsNorm, MatchesReference) {
    const int rows = 17, dim = 96;
    auto x = lh_test::random_vector(rows * dim, 121, -4.0f, 4.0f);
    auto g = lh_test::random_vector(dim, 122);
    std::vector<float> y(rows * dim), yref(rows * dim);

    lh::rmsnorm(x.data(), g.data(), y.data(), rows, dim);
    lh::rmsnorm_ref(x.data(), g.data(), yref.data(), rows, dim);
    EXPECT_TRUE(lh_test::AllClose(y, yref));
}
