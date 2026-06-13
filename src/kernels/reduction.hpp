// LonghornAI — reduction primitives over the last dimension.
//
// Each reduces x[rows, dim] to out[rows]. These back the normalization and
// softmax kernels and are useful standalone.
#ifndef LONGHORNAI_KERNELS_REDUCTION_HPP
#define LONGHORNAI_KERNELS_REDUCTION_HPP

namespace lh {

void reduce_sum(const float* x, float* out, int rows, int dim);
void reduce_max(const float* x, float* out, int rows, int dim);
void reduce_mean(const float* x, float* out, int rows, int dim);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_REDUCTION_HPP
