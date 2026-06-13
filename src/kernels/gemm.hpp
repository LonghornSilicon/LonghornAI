// LonghornAI — general matrix multiply (GEMM) and batched/grouped variants.
//
// Row-major, FP32 accumulate. Each operator ships a dead-simple naive
// reference (`*_ref`) that defines the numerics and a cache-blocked
// implementation that must match it.
#ifndef LONGHORNAI_KERNELS_GEMM_HPP
#define LONGHORNAI_KERNELS_GEMM_HPP

#include <cstdint>

namespace lh {

// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]  (all row-major).
void gemm_ref(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);

// Uniform-shape batched GEMM. A: [batch,M,K], B: [batch,K,N], C: [batch,M,N].
void gemm_batched(const float* A, const float* B, float* C,
                  int batch, int M, int N, int K,
                  float alpha = 1.0f, float beta = 0.0f);

// Variable-shape grouped GEMM. Pointer arrays of length `groups`; group g is
// an (M[g] x K[g]) @ (K[g] x N[g]) product.
void gemm_grouped(const float* const* A, const float* const* B,
                  float* const* C, const int* M, const int* N, const int* K,
                  int groups, float alpha = 1.0f, float beta = 0.0f);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_GEMM_HPP
