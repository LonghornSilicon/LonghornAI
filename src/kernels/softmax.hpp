// LonghornAI — numerically stable row softmax.
#ifndef LONGHORNAI_KERNELS_SOFTMAX_HPP
#define LONGHORNAI_KERNELS_SOFTMAX_HPP

namespace lh {

// Softmax over the last dimension of x[rows, dim] (max-subtracted for
// stability). The reference does the textbook two-pass; the optimized path is
// a single fused pass.
void softmax_ref(const float* x, float* y, int rows, int dim);
void softmax(const float* x, float* y, int rows, int dim);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_SOFTMAX_HPP
