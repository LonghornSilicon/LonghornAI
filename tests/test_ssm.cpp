#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/ssm.hpp"
#include "test_util.hpp"

namespace {

lh::SelectiveScanConfig make_cfg(int B, int L, int Di, int Ds) {
    lh::SelectiveScanConfig c;
    c.batch = B;
    c.seq = L;
    c.d_inner = Di;
    c.d_state = Ds;
    return c;
}

// Random-but-valid Mamba inputs: A is negative (stable continuous-time
// eigenvalues); delta is positive; B/C/x random. D is small.
struct Inputs {
    std::vector<float> x, delta, A, B, C, D;
};

Inputs build_inputs(int B, int L, int Di, int Ds, uint32_t seed) {
    Inputs in;
    in.x = lh_test::random_vector(B * L * Di, seed);
    in.delta = lh_test::random_vector(B * L * Di, seed + 1, 0.05f, 0.5f);
    in.A = lh_test::random_vector(Di * Ds, seed + 2, -2.0f, -0.05f);
    in.B = lh_test::random_vector(B * L * Ds, seed + 3, -1.0f, 1.0f);
    in.C = lh_test::random_vector(B * L * Ds, seed + 4, -1.0f, 1.0f);
    in.D = lh_test::random_vector(Di, seed + 5, -0.1f, 0.1f);
    return in;
}

}  // namespace

TEST(SSM, ChunkedMatchesReferenceForVariousChunkSizes) {
    auto cfg = make_cfg(2, 17, 4, 8);
    auto in = build_inputs(2, 17, 4, 8, 9000);

    std::vector<float> y_ref(static_cast<size_t>(cfg.batch) * cfg.seq *
                             cfg.d_inner);
    lh::selective_scan_ref(in.x.data(), in.delta.data(), in.A.data(),
                           in.B.data(), in.C.data(), in.D.data(), y_ref.data(),
                           cfg);

    for (int chunk : {1, 2, 4, 8, 17, 32}) {
        std::vector<float> y_chunked(y_ref.size(), 0.0f);
        lh::selective_scan_chunked(in.x.data(), in.delta.data(), in.A.data(),
                                   in.B.data(), in.C.data(), in.D.data(),
                                   y_chunked.data(), chunk, cfg);
        EXPECT_TRUE(lh_test::AllClose(y_chunked, y_ref, 1e-4f, 1e-4f))
            << "chunk_size = " << chunk;
    }
}

TEST(SSM, NoSkipConnectionWithNullD) {
    auto cfg = make_cfg(1, 6, 2, 4);
    auto in = build_inputs(1, 6, 2, 4, 9100);
    std::vector<float> y_ref(cfg.seq * cfg.d_inner, 0.0f);
    std::vector<float> y_chunked(cfg.seq * cfg.d_inner, 0.0f);
    lh::selective_scan_ref(in.x.data(), in.delta.data(), in.A.data(),
                           in.B.data(), in.C.data(), nullptr, y_ref.data(),
                           cfg);
    lh::selective_scan_chunked(in.x.data(), in.delta.data(), in.A.data(),
                               in.B.data(), in.C.data(), nullptr,
                               y_chunked.data(), 4, cfg);
    EXPECT_TRUE(lh_test::AllClose(y_chunked, y_ref, 1e-4f, 1e-4f));
}

TEST(SSM, ZeroDeltaProducesZeroState) {
    // delta = 0 → A_bar = 1 (state unchanged), B_bar = 0 (no input).
    // Starting from h_0 = 0 the state stays 0 forever, so y = D * x.
    auto cfg = make_cfg(1, 5, 3, 4);
    auto in = build_inputs(1, 5, 3, 4, 9200);
    std::fill(in.delta.begin(), in.delta.end(), 0.0f);
    std::vector<float> y(cfg.seq * cfg.d_inner, 0.0f);
    lh::selective_scan_ref(in.x.data(), in.delta.data(), in.A.data(),
                           in.B.data(), in.C.data(), in.D.data(), y.data(),
                           cfg);
    for (int t = 0; t < cfg.seq; ++t) {
        for (int d = 0; d < cfg.d_inner; ++d) {
            const float expected =
                in.D[d] * in.x[(t * cfg.d_inner) + d];
            EXPECT_NEAR(y[t * cfg.d_inner + d], expected, 1e-6f);
        }
    }
}
