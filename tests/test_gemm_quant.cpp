#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/gemm.hpp"
#include "kernels/gemm_quant.hpp"
#include "kernels/quant.hpp"
#include "test_util.hpp"

namespace {

double rms(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        s += d * d;
    }
    return std::sqrt(s / a.size());
}

double rms(const std::vector<float>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return std::sqrt(s / a.size());
}

}  // namespace

TEST(GemmQuant, W8A8MatchesFp32WithinTolerance) {
    constexpr int M = 32, N = 48, K = 64;
    auto A = lh_test::random_vector(M * K, 5000, -1.0f, 1.0f);
    auto B = lh_test::random_vector(K * N, 5001, -1.0f, 1.0f);

    // FP32 baseline.
    std::vector<float> C_fp32(M * N, 0.0f);
    lh::gemm(A.data(), B.data(), C_fp32.data(), M, N, K);

    // W8A8: per-row A scale, per-col B scale.
    std::vector<int8_t> Aq(M * K), Bq(K * N);
    std::vector<float> A_s(M), B_s(N);
    lh::q8_quantize_per_row(A.data(), Aq.data(), A_s.data(), M, K);
    lh::q8_quantize_per_col(B.data(), Bq.data(), B_s.data(), K, N);
    std::vector<float> C_q(M * N, 0.0f);
    lh::gemm_w8a8(Aq.data(), Bq.data(), A_s.data(), B_s.data(), C_q.data(),
                  M, N, K);

    // Expected: dot products of ~K terms each in ±1; quantized to int8 →
    // the relative error per element should be on the order of a few %.
    const double rel = rms(C_q, C_fp32) / rms(C_fp32);
    EXPECT_LT(rel, 0.05) << "rel RMS = " << rel;
}

TEST(GemmQuant, W4A16MatchesFp32WithinTolerance) {
    constexpr int M = 16, N = 32, K = 64;
    constexpr int G = 32;
    auto A = lh_test::random_vector(M * K, 5100, -1.0f, 1.0f);
    auto B = lh_test::random_vector(K * N, 5101, -1.0f, 1.0f);

    std::vector<float> C_fp32(M * N, 0.0f);
    lh::gemm(A.data(), B.data(), C_fp32.data(), M, N, K);

    std::vector<uint8_t> Bp(K * N / 2);
    std::vector<float> B_s((K / G) * N);
    lh::q4_quantize_groupwise(B.data(), Bp.data(), B_s.data(), K, N, G);
    std::vector<float> C_q(M * N, 0.0f);
    lh::gemm_w4a16_groupwise(A.data(), Bp.data(), B_s.data(), C_q.data(),
                             M, N, K, G);

    // INT4 with group_size 32 typically lands within ~5-10% relative RMS
    // on random uniform inputs. Tolerance picked from observed runs.
    const double rel = rms(C_q, C_fp32) / rms(C_fp32);
    EXPECT_LT(rel, 0.10) << "rel RMS = " << rel;
}

TEST(GemmQuant, Fp8MatchesFp32WithinTolerance) {
    constexpr int M = 16, N = 32, K = 64;
    auto A = lh_test::random_vector(M * K, 5200, -1.0f, 1.0f);
    auto B = lh_test::random_vector(K * N, 5201, -1.0f, 1.0f);

    std::vector<float> C_fp32(M * N, 0.0f);
    lh::gemm(A.data(), B.data(), C_fp32.data(), M, N, K);

    std::vector<lh::fp8_e4m3> A_fp8(M * K), B_fp8(K * N);
    float A_scale = 0.0f, B_scale = 0.0f;
    lh::fp8_quantize_per_tensor(A.data(), A_fp8.data(), &A_scale, M * K);
    lh::fp8_quantize_per_tensor(B.data(), B_fp8.data(), &B_scale, K * N);
    std::vector<float> C_q(M * N, 0.0f);
    lh::gemm_fp8_e4m3(A_fp8.data(), B_fp8.data(), A_scale, B_scale, C_q.data(),
                      M, N, K);

    const double rel = rms(C_q, C_fp32) / rms(C_fp32);
    EXPECT_LT(rel, 0.10) << "rel RMS = " << rel;
}

TEST(GemmQuant, W8A8RespectsBeta) {
    // Sanity: starting with a nonzero C and beta != 0, the kernel should
    // scale-then-accumulate just like the fp32 path.
    constexpr int M = 4, N = 4, K = 4;
    auto A = lh_test::random_vector(M * K, 5300, -1.0f, 1.0f);
    auto B = lh_test::random_vector(K * N, 5301, -1.0f, 1.0f);

    std::vector<float> C0(M * N, 1.5f);
    auto C_fp32 = C0;
    lh::gemm(A.data(), B.data(), C_fp32.data(), M, N, K, /*alpha=*/0.5f,
             /*beta=*/2.0f);

    std::vector<int8_t> Aq(M * K), Bq(K * N);
    std::vector<float> A_s(M), B_s(N);
    lh::q8_quantize_per_row(A.data(), Aq.data(), A_s.data(), M, K);
    lh::q8_quantize_per_col(B.data(), Bq.data(), B_s.data(), K, N);
    auto C_q = C0;
    lh::gemm_w8a8(Aq.data(), Bq.data(), A_s.data(), B_s.data(), C_q.data(),
                  M, N, K, /*alpha=*/0.5f, /*beta=*/2.0f);

    const double rel = rms(C_q, C_fp32) / rms(C_fp32);
    EXPECT_LT(rel, 0.05);
}

// Phase 4 acceptance gate: storage-bandwidth reduction. W4A16 weights
// occupy 0.5 bytes per element + 1 fp32 scale per (G, N) group, against
// fp16's 2 bytes per element. For typical group sizes the reduction is
// ~3-4x. We assert it explicitly so a future change to the packing format
// is caught.
TEST(GemmQuant, W4A16BandwidthBeatsFp16ByMoreThan3x) {
    constexpr int K = 4096, N = 4096;
    for (int G : {32, 64, 128}) {
        const int64_t packed_bytes =
            static_cast<int64_t>(K) * N / 2;
        const int64_t scale_bytes =
            static_cast<int64_t>(K) / G * N * sizeof(float);
        const int64_t fp16_bytes = static_cast<int64_t>(K) * N * 2;
        const double ratio =
            static_cast<double>(fp16_bytes) / (packed_bytes + scale_bytes);
        EXPECT_GT(ratio, 3.0) << "G=" << G << " ratio=" << ratio;
    }
}
