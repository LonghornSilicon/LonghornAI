#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/softmax.hpp"
#include "test_util.hpp"

TEST(Softmax, MatchesReference) {
    const int rows = 23, dim = 129;
    auto x = lh_test::random_vector(rows * dim, 301, -10.0f, 10.0f);
    std::vector<float> y(rows * dim), yref(rows * dim);
    lh::softmax(x.data(), y.data(), rows, dim);
    lh::softmax_ref(x.data(), yref.data(), rows, dim);
    EXPECT_TRUE(lh_test::AllClose(y, yref));
}

TEST(Softmax, RowsSumToOne) {
    const int rows = 8, dim = 100;
    auto x = lh_test::random_vector(rows * dim, 302, -5.0f, 5.0f);
    std::vector<float> y(rows * dim);
    lh::softmax(x.data(), y.data(), rows, dim);
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int i = 0; i < dim; ++i) s += y[r * dim + i];
        EXPECT_NEAR(s, 1.0, 1e-5);
    }
}

TEST(Softmax, StableForLargeInputs) {
    // Large magnitudes must not overflow thanks to max-subtraction.
    std::vector<float> x = {1000.0f, 1001.0f, 999.0f, 1000.5f};
    std::vector<float> y(x.size());
    lh::softmax(x.data(), y.data(), 1, static_cast<int>(x.size()));
    double s = 0.0;
    for (float v : y) {
        EXPECT_FALSE(std::isnan(v));
        s += v;
    }
    EXPECT_NEAR(s, 1.0, 1e-5);
}
