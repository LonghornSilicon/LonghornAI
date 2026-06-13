#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "kernels/reduction.hpp"
#include "test_util.hpp"

TEST(Reduction, SumMaxMean) {
    const int rows = 13, dim = 64;
    auto x = lh_test::random_vector(rows * dim, 401, -3.0f, 3.0f);
    std::vector<float> s(rows), mx(rows), mn(rows);
    lh::reduce_sum(x.data(), s.data(), rows, dim);
    lh::reduce_max(x.data(), mx.data(), rows, dim);
    lh::reduce_mean(x.data(), mn.data(), rows, dim);

    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        float maxv = x[r * dim];
        for (int i = 0; i < dim; ++i) {
            sum += x[r * dim + i];
            maxv = std::max(maxv, x[r * dim + i]);
        }
        EXPECT_NEAR(s[r], static_cast<float>(sum), 1e-3f);
        EXPECT_NEAR(mx[r], maxv, 1e-6f);
        EXPECT_NEAR(mn[r], static_cast<float>(sum / dim), 1e-4f);
    }
}
