#include "kernels/reduction.hpp"

#include <cstdint>
#include <limits>

namespace lh {

void reduce_sum(const float* x, float* out, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float s = 0.0f;
        for (int i = 0; i < dim; ++i) s += xr[i];
        out[r] = s;
    }
}

void reduce_max(const float* x, float* out, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < dim; ++i) m = (xr[i] > m) ? xr[i] : m;
        out[r] = m;
    }
}

void reduce_mean(const float* x, float* out, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float s = 0.0f;
        for (int i = 0; i < dim; ++i) s += xr[i];
        out[r] = (dim > 0) ? s / static_cast<float>(dim) : 0.0f;
    }
}

}  // namespace lh
