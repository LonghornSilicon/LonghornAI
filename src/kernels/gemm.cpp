#include "kernels/gemm.hpp"

namespace lh {

void gemm_ref(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

// Cache-blocked GEMM. The inner kernel walks K in the middle loop with the
// j (N) loop innermost so both B and C are streamed contiguously, which keeps
// the hot rows of C resident and the B accesses coalesced.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, float alpha, float beta) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

    // Apply beta up front so the accumulation phase is pure add.
    for (int i = 0; i < M; ++i) {
        float* crow = C + static_cast<int64_t>(i) * N;
        if (beta == 0.0f) {
            for (int j = 0; j < N; ++j) crow[j] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; ++j) crow[j] *= beta;
        }
    }

    for (int i0 = 0; i0 < M; i0 += BM) {
        const int imax = (i0 + BM < M) ? i0 + BM : M;
        for (int k0 = 0; k0 < K; k0 += BK) {
            const int kmax = (k0 + BK < K) ? k0 + BK : K;
            for (int j0 = 0; j0 < N; j0 += BN) {
                const int jmax = (j0 + BN < N) ? j0 + BN : N;
                for (int i = i0; i < imax; ++i) {
                    const float* arow = A + static_cast<int64_t>(i) * K;
                    float* crow = C + static_cast<int64_t>(i) * N;
                    for (int k = k0; k < kmax; ++k) {
                        const float a = alpha * arow[k];
                        const float* brow = B + static_cast<int64_t>(k) * N;
                        for (int j = j0; j < jmax; ++j) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_batched(const float* A, const float* B, float* C,
                  int batch, int M, int N, int K, float alpha, float beta) {
    const int64_t as = static_cast<int64_t>(M) * K;
    const int64_t bs = static_cast<int64_t>(K) * N;
    const int64_t cs = static_cast<int64_t>(M) * N;
    for (int b = 0; b < batch; ++b) {
        gemm(A + b * as, B + b * bs, C + b * cs, M, N, K, alpha, beta);
    }
}

void gemm_grouped(const float* const* A, const float* const* B,
                  float* const* C, const int* M, const int* N, const int* K,
                  int groups, float alpha, float beta) {
    for (int g = 0; g < groups; ++g) {
        gemm(A[g], B[g], C[g], M[g], N[g], K[g], alpha, beta);
    }
}

}  // namespace lh
