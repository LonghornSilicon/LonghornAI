// LonghornAI — quantized GEMM kernels.
//
// All variants compute the same operation as `gemm`:
//   C[M, N] = alpha * A[M, K] @ B[K, N] + beta * C[M, N]
// with one or both operands in a low-precision dtype. Accumulation is
// always FP32 — this is a hard silicon requirement for numerical stability.
// Outputs are FP32 (callers that want low-precision output requantize
// downstream).
//
// The naming convention "WxAy" follows the field default: W = weights (B),
// A = activations (A), and the numbers are bit-widths.
//
//   - gemm_w8a8: INT8 weights × INT8 activations, INT32 accumulate,
//     fp32 rescale at the epilogue with per-row (activation) and
//     per-column (weight) scales.
//   - gemm_w4a16_groupwise: INT4-group weights × FP32 activations, on-the-
//     fly dequantization on the read path. The standard W4A16 path for
//     bandwidth-constrained decode.
//   - gemm_fp8_e4m3: FP8 (E4M3) × FP8 (E4M3), fp32 accumulate, per-tensor
//     scales.
//
// All kernels are correctness references first; performance follows from
// fusing dequant/rescale into the inner loop, but the algorithmic shape
// here is the one a tensor unit + rescale unit will execute.
#ifndef LONGHORNAI_KERNELS_GEMM_QUANT_HPP
#define LONGHORNAI_KERNELS_GEMM_QUANT_HPP

#include <cstdint>

#include "kernels/dtypes.hpp"

namespace lh {

// W8A8 GEMM. Both operands are INT8.
//   A_q: int8 [M, K] (activations), per-row scale `A_scales` length M.
//   B_q: int8 [K, N] (weights), per-col scale `B_scales` length N.
//   C:   fp32 [M, N] output.
//
//   C[m, n] = beta * C[m, n] +
//             alpha * A_scales[m] * B_scales[n] *
//                     sum_k A_q[m, k] * B_q[k, n]   (INT32 accumulate)
void gemm_w8a8(const int8_t* A_q, const int8_t* B_q,
               const float* A_scales, const float* B_scales,
               float* C, int M, int N, int K,
               float alpha = 1.0f, float beta = 0.0f);

// W4A16 (group-quantized weights, FP32 activations) GEMM.
//   A:           fp32 [M, K]
//   B_packed:    uint8 [K * N / 2]  (two int4 per byte, see `quant.hpp`)
//   B_scales:    fp32 [K/G, N]
//   C:           fp32 [M, N]
// `group_size` divides K. Dequantizes one K-group at a time into a
// scratch panel and accumulates into C.
void gemm_w4a16_groupwise(const float* A, const uint8_t* B_packed,
                          const float* B_scales, float* C,
                          int M, int N, int K, int group_size,
                          float alpha = 1.0f, float beta = 0.0f);

// FP8 (E4M3) GEMM with per-tensor scales.
//   A: fp8_e4m3 [M, K], scale `A_scale`
//   B: fp8_e4m3 [K, N], scale `B_scale`
//   C: fp32 [M, N]
//
//   C[m, n] = beta * C[m, n] +
//             alpha * A_scale * B_scale *
//                     sum_k float(A[m, k]) * float(B[k, n])   (fp32 acc)
void gemm_fp8_e4m3(const fp8_e4m3* A, const fp8_e4m3* B,
                   float A_scale, float B_scale,
                   float* C, int M, int N, int K,
                   float alpha = 1.0f, float beta = 0.0f);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_GEMM_QUANT_HPP
