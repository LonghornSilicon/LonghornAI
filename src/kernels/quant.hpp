// LonghornAI — quantization utilities.
//
// Float ↔ low-precision conversions for the dtypes that appear in modern
// inference: INT8, packed INT4 (group-quantized), FP8 (E4M3). The
// quantization *scheme* (symmetric vs. asymmetric, scope of the scale) is
// orthogonal to the *storage* dtype; this header parameterises both.
//
// Naming convention:
//   q8_*    -> INT8 storage
//   q4_*    -> packed INT4 storage (two values per byte)
//   fp8_*   -> FP8 E4M3 storage
//
// Scales are always FP32 — accumulator precision is sacred. Per-row /
// per-channel scopes are passed as separate scale buffers; per-group is
// indexed by `(row_or_col, group_idx)`.
#ifndef LONGHORNAI_KERNELS_QUANT_HPP
#define LONGHORNAI_KERNELS_QUANT_HPP

#include <cstddef>
#include <cstdint>

#include "kernels/dtypes.hpp"

namespace lh {

// --- INT8 symmetric -------------------------------------------------------

// Per-tensor: one fp32 scale for the whole buffer.
//   scale = max(abs(x)) / 127
//   x_q   = round(x / scale)    (clamped to [-127, 127])
void q8_quantize_per_tensor(const float* src, int8_t* dst, float* scale_out,
                            int64_t n);
void q8_dequantize_per_tensor(const int8_t* src, float scale, float* dst,
                              int64_t n);

// Per-row (== per-token for activations laid out [tokens, dim]). One scale
// per row of length `dim`. `scales_out` has length `rows`.
void q8_quantize_per_row(const float* src, int8_t* dst, float* scales_out,
                         int rows, int dim);
void q8_dequantize_per_row(const int8_t* src, const float* scales, float* dst,
                           int rows, int dim);

// Per-column (== per-output-channel for weights laid out [in, out]). One
// scale per column of length `cols`. `scales_out` has length `cols`.
void q8_quantize_per_col(const float* src, int8_t* dst, float* scales_out,
                         int rows, int cols);
void q8_dequantize_per_col(const int8_t* src, const float* scales, float* dst,
                           int rows, int cols);

// --- INT4 group (packed) --------------------------------------------------

// Packing convention for [K, N] weights with group-along-K of size `G`:
//   - `packed`:  uint8_t of length (K * N) / 2.
//     Two K-adjacent int4 values share a byte: low 4 bits = row 2*i,
//     high 4 bits = row 2*i + 1, for column n. K must be even.
//   - `scales`:  fp32 of shape [K / G, N] (row-major, group-major along K).
//   - Symmetric quantization: int4 in [-7, 7]; scale = max_abs / 7 per group.
//
// Group size must divide K (typical: 32, 64, 128).
void q4_quantize_groupwise(const float* src, uint8_t* packed,
                           float* scales_out, int K, int N, int group_size);
void q4_dequantize_groupwise(const uint8_t* packed, const float* scales,
                             float* dst, int K, int N, int group_size);

// Unpack a single int4 value from a packed byte. `k` selects the row in
// the [K, N] logical weight tensor; `n` selects the column; `N` is the
// stride between row pairs in the packed buffer.
inline int8_t q4_get(const uint8_t* packed, int k, int n, int N) {
    // Row-major over (K/2, N): byte at (k/2, n) is packed[(k/2) * N + n],
    // low nibble = row 2*i, high nibble = row 2*i + 1.
    const uint8_t b = packed[static_cast<int64_t>(k / 2) * N + n];
    const uint8_t nib = (k & 1) ? (b >> 4) : (b & 0x0fu);
    // Sign-extend from 4 bits to 8.
    return static_cast<int8_t>((nib & 0x08u) ? (nib | 0xf0u) : nib);
}

// --- FP8 E4M3 ------------------------------------------------------------

// Per-tensor scale: x_q = round_e4m3(x / scale). Per-row and per-col
// variants follow the INT8 shape conventions.
void fp8_quantize_per_tensor(const float* src, fp8_e4m3* dst,
                             float* scale_out, int64_t n);
void fp8_dequantize_per_tensor(const fp8_e4m3* src, float scale, float* dst,
                               int64_t n);

void fp8_quantize_per_row(const float* src, fp8_e4m3* dst, float* scales_out,
                          int rows, int dim);
void fp8_dequantize_per_row(const fp8_e4m3* src, const float* scales,
                            float* dst, int rows, int dim);

void fp8_quantize_per_col(const float* src, fp8_e4m3* dst, float* scales_out,
                          int rows, int cols);
void fp8_dequantize_per_col(const fp8_e4m3* src, const float* scales,
                            float* dst, int rows, int cols);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_QUANT_HPP
