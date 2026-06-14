#include "kernels/gemm_quant.hpp"

#include <cstdint>
#include <vector>

#include "kernels/quant.hpp"

namespace lh {

namespace {

inline void apply_beta(float* C, int M, int N, float beta) {
    for (int i = 0; i < M; ++i) {
        float* row = C + static_cast<int64_t>(i) * N;
        if (beta == 0.0f) {
            for (int j = 0; j < N; ++j) row[j] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; ++j) row[j] *= beta;
        }
    }
}

}  // namespace

// INT8 inputs, INT32 accumulator, fp32 rescale at the epilogue with
// per-row × per-col scales. This is the canonical W8A8 shape: the MAC
// array sees uniform int8 inputs, the accumulator is precise, and the
// rescale step is a cheap pointwise multiply at the end.
void gemm_w8a8(const int8_t* A_q, const int8_t* B_q, const float* A_scales,
               const float* B_scales, float* C, int M, int N, int K,
               float alpha, float beta) {
    apply_beta(C, M, N, beta);
    for (int i = 0; i < M; ++i) {
        const int8_t* arow = A_q + static_cast<int64_t>(i) * K;
        const float a_s = A_scales[i];
        float* crow = C + static_cast<int64_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            // Inner k loop: pure integer MACs into a 32-bit accumulator.
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(arow[k]) *
                       static_cast<int32_t>(B_q[static_cast<int64_t>(k) * N + j]);
            }
            const float rescaled =
                alpha * a_s * B_scales[j] * static_cast<float>(acc);
            crow[j] += rescaled;
        }
    }
}

// W4A16: dequantize one K-group of B at a time into a fp32 scratch panel
// of shape [G, N], then run a small fp32 GEMM contribution against it.
// This matches the "dequant on the input feeders" silicon pattern.
void gemm_w4a16_groupwise(const float* A, const uint8_t* B_packed,
                          const float* B_scales, float* C, int M, int N, int K,
                          int group_size, float alpha, float beta) {
    apply_beta(C, M, N, beta);
    const int G = group_size;
    std::vector<float> Bdq(static_cast<size_t>(G) * N);

    for (int k0 = 0; k0 < K; k0 += G) {
        // Dequantize the [k0:k0+G, :] panel of B into Bdq.
        const int g = k0 / G;
        for (int kk = 0; kk < G; ++kk) {
            for (int n = 0; n < N; ++n) {
                const int8_t q = q4_get(B_packed, k0 + kk, n, N);
                Bdq[static_cast<int64_t>(kk) * N + n] =
                    static_cast<float>(q) * B_scales[g * N + n];
            }
        }
        // Accumulate A[:, k0:k0+G] @ Bdq into C.
        for (int i = 0; i < M; ++i) {
            const float* arow = A + static_cast<int64_t>(i) * K + k0;
            float* crow = C + static_cast<int64_t>(i) * N;
            for (int kk = 0; kk < G; ++kk) {
                const float a = alpha * arow[kk];
                const float* brow = Bdq.data() + static_cast<int64_t>(kk) * N;
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            }
        }
    }
}

// FP8 GEMM: convert each element to float at use, accumulate in fp32, then
// apply per-tensor rescale. The conversion itself is what would happen
// inside the input-feeder dequant unit on silicon.
void gemm_fp8_e4m3(const fp8_e4m3* A, const fp8_e4m3* B, float A_scale,
                   float B_scale, float* C, int M, int N, int K, float alpha,
                   float beta) {
    apply_beta(C, M, N, beta);
    const float combined = alpha * A_scale * B_scale;
    for (int i = 0; i < M; ++i) {
        const fp8_e4m3* arow = A + static_cast<int64_t>(i) * K;
        float* crow = C + static_cast<int64_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<float>(arow[k]) *
                       static_cast<float>(B[static_cast<int64_t>(k) * N + j]);
            }
            crow[j] += combined * acc;
        }
    }
}

}  // namespace lh
