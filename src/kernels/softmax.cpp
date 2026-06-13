#include "kernels/softmax.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace lh {

void softmax_ref(const float* x, float* y, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float* yr = y + static_cast<int64_t>(r) * dim;

        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < dim; ++i) m = (xr[i] > m) ? xr[i] : m;

        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            yr[i] = std::exp(xr[i] - m);
            sum += yr[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < dim; ++i) yr[i] *= inv;
    }
}

// Single-pass online softmax: track running max and the running normaliser,
// rescaling the partial sum whenever a new max appears.
void softmax(const float* x, float* y, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float* yr = y + static_cast<int64_t>(r) * dim;

        float m = -std::numeric_limits<float>::infinity();
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float v = xr[i];
            if (v > m) {
                sum *= std::exp(m - v);
                m = v;
            }
            sum += std::exp(v - m);
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < dim; ++i) yr[i] = std::exp(xr[i] - m) * inv;
    }
}

}  // namespace lh
