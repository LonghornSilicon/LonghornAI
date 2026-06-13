#include <gtest/gtest.h>

#include <vector>

#include "kernels/gemm.hpp"
#include "test_util.hpp"

TEST(Gemm, MatchesReference) {
    const int M = 65, K = 70, N = 48;  // deliberately non-aligned
    auto A = lh_test::random_vector(M * K, 11);
    auto B = lh_test::random_vector(K * N, 12);
    std::vector<float> C_ref(M * N, 0.0f), C(M * N, 0.0f);

    lh::gemm_ref(A.data(), B.data(), C_ref.data(), M, N, K);
    lh::gemm(A.data(), B.data(), C.data(), M, N, K);

    EXPECT_TRUE(lh_test::AllClose(C, C_ref, 1e-3f, 1e-4f));
}

TEST(Gemm, AlphaBeta) {
    const int M = 32, K = 16, N = 24;
    auto A = lh_test::random_vector(M * K, 21);
    auto B = lh_test::random_vector(K * N, 22);
    auto C0 = lh_test::random_vector(M * N, 23);

    std::vector<float> C_ref = C0, C = C0;
    lh::gemm_ref(A.data(), B.data(), C_ref.data(), M, N, K, 0.75f, -0.5f);
    lh::gemm(A.data(), B.data(), C.data(), M, N, K, 0.75f, -0.5f);

    EXPECT_TRUE(lh_test::AllClose(C, C_ref, 1e-3f, 1e-4f));
}

TEST(Gemm, Batched) {
    const int batch = 4, M = 12, K = 20, N = 9;
    auto A = lh_test::random_vector(batch * M * K, 31);
    auto B = lh_test::random_vector(batch * K * N, 32);
    std::vector<float> C(batch * M * N, 0.0f), C_ref(batch * M * N, 0.0f);

    lh::gemm_batched(A.data(), B.data(), C.data(), batch, M, N, K);
    for (int b = 0; b < batch; ++b) {
        lh::gemm_ref(A.data() + b * M * K, B.data() + b * K * N,
                     C_ref.data() + b * M * N, M, N, K);
    }
    EXPECT_TRUE(lh_test::AllClose(C, C_ref, 1e-3f, 1e-4f));
}

TEST(Gemm, Grouped) {
    const int M[2] = {8, 5};
    const int K[2] = {6, 9};
    const int N[2] = {7, 4};
    auto A0 = lh_test::random_vector(M[0] * K[0], 41);
    auto A1 = lh_test::random_vector(M[1] * K[1], 42);
    auto B0 = lh_test::random_vector(K[0] * N[0], 43);
    auto B1 = lh_test::random_vector(K[1] * N[1], 44);
    std::vector<float> C0(M[0] * N[0], 0.0f), C1(M[1] * N[1], 0.0f);
    std::vector<float> R0(M[0] * N[0], 0.0f), R1(M[1] * N[1], 0.0f);

    const float* A[2] = {A0.data(), A1.data()};
    const float* B[2] = {B0.data(), B1.data()};
    float* C[2] = {C0.data(), C1.data()};
    lh::gemm_grouped(A, B, C, M, N, K, 2);

    lh::gemm_ref(A0.data(), B0.data(), R0.data(), M[0], N[0], K[0]);
    lh::gemm_ref(A1.data(), B1.data(), R1.data(), M[1], N[1], K[1]);
    EXPECT_TRUE(lh_test::AllClose(C0, R0, 1e-3f, 1e-4f));
    EXPECT_TRUE(lh_test::AllClose(C1, R1, 1e-3f, 1e-4f));
}
