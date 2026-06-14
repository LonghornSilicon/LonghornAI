#include "kernels/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace lh {

namespace {

inline int8_t sat_round_i8(float v, int max_abs = 127) {
    if (v >= max_abs) return static_cast<int8_t>(max_abs);
    if (v <= -max_abs) return static_cast<int8_t>(-max_abs);
    int r = static_cast<int>(std::lrintf(v));
    if (r > max_abs) r = max_abs;
    if (r < -max_abs) r = -max_abs;
    return static_cast<int8_t>(r);
}

inline float row_max_abs(const float* x, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > m) m = a;
    }
    return m;
}

}  // namespace

// --- INT8 symmetric -------------------------------------------------------

void q8_quantize_per_tensor(const float* src, int8_t* dst, float* scale_out,
                            int64_t n) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    const float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int64_t i = 0; i < n; ++i) dst[i] = sat_round_i8(src[i] * inv);
    *scale_out = scale;
}

void q8_dequantize_per_tensor(const int8_t* src, float scale, float* dst,
                              int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = src[i] * scale;
}

void q8_quantize_per_row(const float* src, int8_t* dst, float* scales_out,
                         int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * dim;
        int8_t* drow = dst + static_cast<int64_t>(r) * dim;
        const float m = row_max_abs(row, dim);
        const float scale = (m > 0.0f) ? m / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        for (int i = 0; i < dim; ++i) drow[i] = sat_round_i8(row[i] * inv);
        scales_out[r] = scale;
    }
}

void q8_dequantize_per_row(const int8_t* src, const float* scales, float* dst,
                           int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float s = scales[r];
        const int8_t* sr = src + static_cast<int64_t>(r) * dim;
        float* dr = dst + static_cast<int64_t>(r) * dim;
        for (int i = 0; i < dim; ++i) dr[i] = sr[i] * s;
    }
}

void q8_quantize_per_col(const float* src, int8_t* dst, float* scales_out,
                         int rows, int cols) {
    // Two-pass: column max-abs first (one scan), then quantize.
    for (int c = 0; c < cols; ++c) scales_out[c] = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            const float a = std::fabs(row[c]);
            if (a > scales_out[c]) scales_out[c] = a;
        }
    }
    for (int c = 0; c < cols; ++c) {
        scales_out[c] = (scales_out[c] > 0.0f) ? scales_out[c] / 127.0f : 1.0f;
    }
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * cols;
        int8_t* drow = dst + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            drow[c] = sat_round_i8(row[c] / scales_out[c]);
        }
    }
}

void q8_dequantize_per_col(const int8_t* src, const float* scales, float* dst,
                           int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const int8_t* srow = src + static_cast<int64_t>(r) * cols;
        float* drow = dst + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) drow[c] = srow[c] * scales[c];
    }
}

// --- INT4 group (packed) --------------------------------------------------

namespace {

inline uint8_t pack_nibbles(int8_t lo, int8_t hi) {
    const uint8_t lo_n = static_cast<uint8_t>(lo & 0x0f);
    const uint8_t hi_n = static_cast<uint8_t>(hi & 0x0f);
    return static_cast<uint8_t>(lo_n | (hi_n << 4));
}

inline int8_t sat_round_i4(float v) {
    if (v >= 7.0f) return 7;
    if (v <= -7.0f) return -7;
    int r = static_cast<int>(std::lrintf(v));
    if (r > 7) r = 7;
    if (r < -7) r = -7;
    return static_cast<int8_t>(r);
}

}  // namespace

void q4_quantize_groupwise(const float* src, uint8_t* packed,
                           float* scales_out, int K, int N, int group_size) {
    const int G = group_size;
    const int n_groups = K / G;
    // Per-group max-abs along K, per column N.
    for (int g = 0; g < n_groups; ++g) {
        const int k0 = g * G;
        for (int n = 0; n < N; ++n) {
            float m = 0.0f;
            for (int kk = 0; kk < G; ++kk) {
                const float a =
                    std::fabs(src[(static_cast<int64_t>(k0 + kk) * N) + n]);
                if (a > m) m = a;
            }
            scales_out[g * N + n] = (m > 0.0f) ? m / 7.0f : 1.0f;
        }
    }
    // Quantize K row-pairs at a time into packed bytes.
    for (int k = 0; k < K; k += 2) {
        const int g = k / G;
        for (int n = 0; n < N; ++n) {
            const float s_lo = scales_out[g * N + n];
            const float s_hi = scales_out[((k + 1) / G) * N + n];
            const float v_lo =
                src[(static_cast<int64_t>(k) * N) + n] / s_lo;
            const float v_hi =
                src[(static_cast<int64_t>(k + 1) * N) + n] / s_hi;
            const int8_t lo = sat_round_i4(v_lo);
            const int8_t hi = sat_round_i4(v_hi);
            packed[static_cast<int64_t>(k / 2) * N + n] = pack_nibbles(lo, hi);
        }
    }
}

void q4_dequantize_groupwise(const uint8_t* packed, const float* scales,
                             float* dst, int K, int N, int group_size) {
    const int G = group_size;
    for (int k = 0; k < K; ++k) {
        const int g = k / G;
        for (int n = 0; n < N; ++n) {
            const int8_t q = q4_get(packed, k, n, N);
            dst[static_cast<int64_t>(k) * N + n] = q * scales[g * N + n];
        }
    }
}

// --- FP8 E4M3 ------------------------------------------------------------

void fp8_quantize_per_tensor(const float* src, fp8_e4m3* dst,
                             float* scale_out, int64_t n) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    // E4M3 max-finite is 448. Scale so the largest input maps near 448.
    const float scale = (max_abs > 0.0f) ? max_abs / 448.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int64_t i = 0; i < n; ++i) dst[i] = fp8_e4m3(src[i] * inv);
    *scale_out = scale;
}

void fp8_dequantize_per_tensor(const fp8_e4m3* src, float scale, float* dst,
                               int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]) * scale;
}

void fp8_quantize_per_row(const float* src, fp8_e4m3* dst, float* scales_out,
                          int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * dim;
        fp8_e4m3* drow = dst + static_cast<int64_t>(r) * dim;
        const float m = row_max_abs(row, dim);
        const float scale = (m > 0.0f) ? m / 448.0f : 1.0f;
        const float inv = 1.0f / scale;
        for (int i = 0; i < dim; ++i) drow[i] = fp8_e4m3(row[i] * inv);
        scales_out[r] = scale;
    }
}

void fp8_dequantize_per_row(const fp8_e4m3* src, const float* scales,
                            float* dst, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float s = scales[r];
        const fp8_e4m3* sr = src + static_cast<int64_t>(r) * dim;
        float* dr = dst + static_cast<int64_t>(r) * dim;
        for (int i = 0; i < dim; ++i) dr[i] = static_cast<float>(sr[i]) * s;
    }
}

void fp8_quantize_per_col(const float* src, fp8_e4m3* dst, float* scales_out,
                          int rows, int cols) {
    for (int c = 0; c < cols; ++c) scales_out[c] = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            const float a = std::fabs(row[c]);
            if (a > scales_out[c]) scales_out[c] = a;
        }
    }
    for (int c = 0; c < cols; ++c) {
        scales_out[c] = (scales_out[c] > 0.0f) ? scales_out[c] / 448.0f : 1.0f;
    }
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<int64_t>(r) * cols;
        fp8_e4m3* drow = dst + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            drow[c] = fp8_e4m3(row[c] / scales_out[c]);
        }
    }
}

void fp8_dequantize_per_col(const fp8_e4m3* src, const float* scales,
                            float* dst, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const fp8_e4m3* srow = src + static_cast<int64_t>(r) * cols;
        float* drow = dst + static_cast<int64_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            drow[c] = static_cast<float>(srow[c]) * scales[c];
        }
    }
}

}  // namespace lh
