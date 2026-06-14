// LonghornAI — reduction primitives over the last dimension.
//
// Each reduces x[rows, dim] to out[rows]. These back the normalization and
// softmax kernels and are useful standalone.
#ifndef LONGHORNAI_KERNELS_REDUCTION_HPP
#define LONGHORNAI_KERNELS_REDUCTION_HPP

#include <cstdint>

namespace lh {

void reduce_sum(const float* x, float* out, int rows, int dim);
void reduce_max(const float* x, float* out, int rows, int dim);
void reduce_mean(const float* x, float* out, int rows, int dim);

// Index of the per-row max. Ties resolved by lowest index. `out_idx` is
// length `rows`; supply `out_val` to also receive the max value.
void argmax(const float* x, int32_t* out_idx, float* out_val,
            int rows, int dim);

// Top-k per row, descending by value. `out_val` and `out_idx` are
// `[rows, k]` (row-major); ties resolved by lowest index. Output is in
// sorted-descending order, which is the form sampling kernels want.
void topk(const float* x, float* out_val, int32_t* out_idx,
          int rows, int dim, int k);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_REDUCTION_HPP
