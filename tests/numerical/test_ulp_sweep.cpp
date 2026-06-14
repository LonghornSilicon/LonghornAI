// LonghornAI — ULP/numerical sweep over the existing kernel set.
//
// For each kernel that ships both a `*_ref` and an optimized path, this
// suite computes ULP and absolute/relative error distributions and asserts
// they fall within the per-dtype tolerance policy in `numerical/ulp.hpp`.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "kernels/gemm.hpp"
#include "kernels/normalization.hpp"
#include "kernels/rope.hpp"
#include "kernels/softmax.hpp"
#include "numerical/ulp.hpp"

namespace {

std::vector<float> rand_f32(int64_t n, uint32_t seed, float lo = -1.0f,
                            float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

}  // namespace

TEST(Numerical, GemmOptimizedVsRef) {
    constexpr int M = 64, N = 96, K = 80;
    auto A = rand_f32(M * K, 1);
    auto B = rand_f32(K * N, 2);
    std::vector<float> Cref(M * N, 0.0f), Cgot(M * N, 0.0f);
    lh::gemm_ref(A.data(), B.data(), Cref.data(), M, N, K);
    lh::gemm(A.data(), B.data(), Cgot.data(), M, N, K);

    auto rep = lh_num::sweep(Cgot, Cref, lh_num::default_tolerance(lh::DType::F32));
    EXPECT_EQ(rep.n_violations, 0u)
        << "max_ulp=" << rep.max_ulp << " max_abs=" << rep.max_abs_err
        << " max_rel=" << rep.max_rel_err;
    // GEMM through a blocked path should sit within a handful of ULP of the
    // reference for benign random inputs.
    EXPECT_LT(rep.max_ulp, 4096u);
}

TEST(Numerical, RmsNormOptimizedVsRef) {
    constexpr int rows = 32, dim = 512;
    auto x = rand_f32(rows * dim, 5);
    auto g = rand_f32(dim, 6, 0.5f, 1.5f);
    std::vector<float> yref(x.size()), ygot(x.size());
    lh::rmsnorm_ref(x.data(), g.data(), yref.data(), rows, dim);
    lh::rmsnorm(x.data(), g.data(), ygot.data(), rows, dim);

    auto rep = lh_num::sweep(ygot, yref, lh_num::default_tolerance(lh::DType::F32));
    EXPECT_EQ(rep.n_violations, 0u);
}

TEST(Numerical, SoftmaxOptimizedVsRef) {
    constexpr int rows = 16, dim = 1024;
    auto x = rand_f32(rows * dim, 9, -10.0f, 10.0f);
    std::vector<float> yref(x.size()), ygot(x.size());
    lh::softmax_ref(x.data(), yref.data(), rows, dim);
    lh::softmax(x.data(), ygot.data(), rows, dim);

    auto rep = lh_num::sweep(ygot, yref, lh_num::default_tolerance(lh::DType::F32));
    EXPECT_EQ(rep.n_violations, 0u);
}

TEST(Numerical, RopeOptimizedVsRef) {
    constexpr int seq = 17, n_heads = 4, head_dim = 64;
    auto xref = rand_f32(seq * n_heads * head_dim, 21);
    auto xgot = xref;
    lh::rope_ref(xref.data(), seq, n_heads, head_dim);
    lh::rope(xgot.data(), seq, n_heads, head_dim);

    auto rep = lh_num::sweep(xgot, xref, lh_num::default_tolerance(lh::DType::F32));
    EXPECT_EQ(rep.n_violations, 0u);
}

TEST(Numerical, UlpDistanceBasics) {
    EXPECT_EQ(lh_num::ulp_distance_f32(1.0f, 1.0f), 0u);
    EXPECT_EQ(lh_num::ulp_distance_f32(0.0f, -0.0f), 0u);
    // Adjacent FP32 values are exactly 1 ULP apart.
    const float a = 1.0f;
    float b;
    {
        uint32_t ai;
        std::memcpy(&ai, &a, sizeof(ai));
        ++ai;
        std::memcpy(&b, &ai, sizeof(b));
    }
    EXPECT_EQ(lh_num::ulp_distance_f32(a, b), 1u);
}
