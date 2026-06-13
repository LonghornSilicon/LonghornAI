#include "kernels/activation.hpp"

#include <cmath>

namespace lh {

namespace {
constexpr float kSqrt2 = 1.41421356237309504880f;        // sqrt(2)
constexpr float kSqrt2OverPi = 0.79788456080286535588f;  // sqrt(2/pi)
constexpr float kGeluC = 0.044715f;

inline float sigmoid_scalar(float v) { return 1.0f / (1.0f + std::exp(-v)); }

inline float gelu_tanh_scalar(float v) {
    const float inner = kSqrt2OverPi * (v + kGeluC * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(inner));
}
}  // namespace

void gelu_erf(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] / kSqrt2));
    }
}

void gelu_tanh(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = gelu_tanh_scalar(x[i]);
}

void silu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * sigmoid_scalar(x[i]);
}

void sigmoid(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = sigmoid_scalar(x[i]);
}

void swiglu(const float* x, float* y, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* gate = x + static_cast<int64_t>(r) * 2 * dim;
        const float* value = gate + dim;
        float* yr = y + static_cast<int64_t>(r) * dim;
        for (int i = 0; i < dim; ++i) {
            yr[i] = (gate[i] * sigmoid_scalar(gate[i])) * value[i];
        }
    }
}

void geglu(const float* x, float* y, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* gate = x + static_cast<int64_t>(r) * 2 * dim;
        const float* value = gate + dim;
        float* yr = y + static_cast<int64_t>(r) * dim;
        for (int i = 0; i < dim; ++i) {
            yr[i] = gelu_tanh_scalar(gate[i]) * value[i];
        }
    }
}

}  // namespace lh
