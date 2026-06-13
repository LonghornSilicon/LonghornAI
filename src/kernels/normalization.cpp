#include "kernels/normalization.hpp"

#include <cmath>
#include <cstdint>

namespace lh {

void layernorm_ref(const float* x, const float* gamma, const float* beta,
                   float* y, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float* yr = y + static_cast<int64_t>(r) * dim;

        // Welford one-pass mean/variance.
        float mean = 0.0f;
        float m2 = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float delta = xr[i] - mean;
            mean += delta / static_cast<float>(i + 1);
            m2 += delta * (xr[i] - mean);
        }
        const float var = (dim > 0) ? m2 / static_cast<float>(dim) : 0.0f;
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < dim; ++i) {
            float v = (xr[i] - mean) * inv_std * gamma[i];
            if (beta) v += beta[i];
            yr[i] = v;
        }
    }
}

void layernorm(const float* x, const float* gamma, const float* beta,
               float* y, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float* yr = y + static_cast<int64_t>(r) * dim;

        float sum = 0.0f;
        float sumsq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += xr[i];
            sumsq += xr[i] * xr[i];
        }
        const float inv_n = 1.0f / static_cast<float>(dim);
        const float mean = sum * inv_n;
        const float var = sumsq * inv_n - mean * mean;
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < dim; ++i) {
            float v = (xr[i] - mean) * inv_std * gamma[i];
            if (beta) v += beta[i];
            yr[i] = v;
        }
    }
}

void rmsnorm_ref(const float* x, const float* gamma, float* y,
                 int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float* yr = y + static_cast<int64_t>(r) * dim;

        float sumsq = 0.0f;
        for (int i = 0; i < dim; ++i) sumsq += xr[i] * xr[i];
        const float inv_rms =
            1.0f / std::sqrt(sumsq / static_cast<float>(dim) + eps);

        for (int i = 0; i < dim; ++i) yr[i] = xr[i] * inv_rms * gamma[i];
    }
}

void rmsnorm(const float* x, const float* gamma, float* y,
             int rows, int dim, float eps) {
    rmsnorm_ref(x, gamma, y, rows, dim, eps);
}

}  // namespace lh
