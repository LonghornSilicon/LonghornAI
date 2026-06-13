#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/activation.hpp"
#include "test_util.hpp"

TEST(Activation, GeluErfKnownValues) {
    std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> y(x.size());
    lh::gelu_erf(x.data(), y.data(), static_cast<int64_t>(x.size()));
    // Reference values of exact GELU.
    EXPECT_NEAR(y[0], 0.0f, 1e-5f);
    EXPECT_NEAR(y[1], 0.8413447f, 1e-4f);
    EXPECT_NEAR(y[2], -0.1586553f, 1e-4f);
    EXPECT_NEAR(y[3], 1.9544997f, 1e-4f);
}

TEST(Activation, GeluTanhCloseToErf) {
    auto x = lh_test::random_vector(2048, 201, -6.0f, 6.0f);
    std::vector<float> ye(x.size()), yt(x.size());
    lh::gelu_erf(x.data(), ye.data(), static_cast<int64_t>(x.size()));
    lh::gelu_tanh(x.data(), yt.data(), static_cast<int64_t>(x.size()));
    // The tanh approximation tracks exact GELU to a few thousandths.
    EXPECT_TRUE(lh_test::AllClose(yt, ye, 5e-3f, 5e-3f));
}

TEST(Activation, SiluMatchesFormula) {
    auto x = lh_test::random_vector(1024, 202, -8.0f, 8.0f);
    std::vector<float> y(x.size());
    lh::silu(x.data(), y.data(), static_cast<int64_t>(x.size()));
    for (size_t i = 0; i < x.size(); ++i) {
        const float expected = x[i] / (1.0f + std::exp(-x[i]));
        EXPECT_NEAR(y[i], expected, 1e-5f);
    }
}

TEST(Activation, SwiGLU) {
    const int rows = 7, dim = 32;
    auto x = lh_test::random_vector(rows * 2 * dim, 203, -4.0f, 4.0f);
    std::vector<float> y(rows * dim);
    lh::swiglu(x.data(), y.data(), rows, dim);
    for (int r = 0; r < rows; ++r) {
        const float* gate = x.data() + r * 2 * dim;
        const float* val = gate + dim;
        for (int i = 0; i < dim; ++i) {
            const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
            EXPECT_NEAR(y[r * dim + i], silu * val[i], 1e-5f);
        }
    }
}

TEST(Activation, GeGLU) {
    const int rows = 5, dim = 16;
    auto x = lh_test::random_vector(rows * 2 * dim, 204, -4.0f, 4.0f);
    std::vector<float> y(rows * dim), gate_act(rows * dim);
    lh::geglu(x.data(), y.data(), rows, dim);
    for (int r = 0; r < rows; ++r) {
        const float* gate = x.data() + r * 2 * dim;
        const float* val = gate + dim;
        std::vector<float> ga(dim);
        lh::gelu_tanh(gate, ga.data(), dim);
        for (int i = 0; i < dim; ++i) {
            EXPECT_NEAR(y[r * dim + i], ga[i] * val[i], 1e-5f);
        }
    }
}
