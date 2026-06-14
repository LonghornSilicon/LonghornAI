// LonghornAI — activation kernels.
//
// Pointwise activations plus the gated-MLP variants (SwiGLU / GeGLU) used by
// Llama-family feed-forward blocks.
#ifndef LONGHORNAI_KERNELS_ACTIVATION_HPP
#define LONGHORNAI_KERNELS_ACTIVATION_HPP

#include <cstdint>

namespace lh {

// Exact GELU using the error function.
void gelu_erf(const float* x, float* y, int64_t n);

// tanh-approximation GELU (the variant most LLMs ship).
void gelu_tanh(const float* x, float* y, int64_t n);

// SiLU / swish: x * sigmoid(x).
void silu(const float* x, float* y, int64_t n);

// sigmoid: 1 / (1 + e^-x).
void sigmoid(const float* x, float* y, int64_t n);

// ReLU: max(0, x).
void relu(const float* x, float* y, int64_t n);

// tanh.
void tanh_act(const float* x, float* y, int64_t n);

// Gated activations. Input x is [rows, 2*dim]; the first `dim` columns are the
// gate, the second `dim` the value. Output y is [rows, dim].
//   SwiGLU: y = silu(gate) * value
//   GeGLU:  y = gelu_tanh(gate) * value
void swiglu(const float* x, float* y, int rows, int dim);
void geglu(const float* x, float* y, int rows, int dim);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_ACTIVATION_HPP
