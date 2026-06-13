// LonghornAI — normalization kernels: LayerNorm and RMSNorm.
//
// Both operate row-wise on x[rows, dim]; reductions accumulate in FP32.
#ifndef LONGHORNAI_KERNELS_NORMALIZATION_HPP
#define LONGHORNAI_KERNELS_NORMALIZATION_HPP

namespace lh {

// LayerNorm over the last dimension. gamma/beta are length `dim` (beta may be
// null for no shift). Reference uses Welford for a stable mean/variance.
void layernorm_ref(const float* x, const float* gamma, const float* beta,
                   float* y, int rows, int dim, float eps = 1e-5f);

void layernorm(const float* x, const float* gamma, const float* beta,
               float* y, int rows, int dim, float eps = 1e-5f);

// RMSNorm over the last dimension. gamma is length `dim`.
void rmsnorm_ref(const float* x, const float* gamma, float* y,
                 int rows, int dim, float eps = 1e-5f);

void rmsnorm(const float* x, const float* gamma, float* y,
             int rows, int dim, float eps = 1e-5f);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_NORMALIZATION_HPP
