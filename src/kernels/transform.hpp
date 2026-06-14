// LonghornAI — tensor transforms.
//
// Layout-only operations (transpose, permute, concat, split) and
// shape-changing index ops (gather, scatter). Most of these are pure data
// motion at FP32; on accelerator silicon they would be pushed onto the DMA
// engine. The reference implementations live here as the correctness
// oracles.
//
// Shapes are passed as raw int64_t arrays plus a rank, to keep the kernel
// signatures vocabulary-free (no `std::vector` in headers). Strides are
// always row-major and inferred from shape.
#ifndef LONGHORNAI_KERNELS_TRANSFORM_HPP
#define LONGHORNAI_KERNELS_TRANSFORM_HPP

#include <cstdint>

namespace lh {

// 2D transpose: y[N, M] = x[M, N].
void transpose2d(const float* x, float* y, int M, int N);

// N-D permute. Input has rank `rank` and shape `shape`. `perm` is a
// permutation of [0, rank). Output shape is shape[perm[i]]; output strides
// are row-major over the permuted shape.
void permute(const float* x, float* y, const int64_t* shape,
             const int* perm, int rank);

// Concatenate `n` tensors along `axis`. All inputs must share shape on
// every other axis. `xs[i]` has shape `shape_in[i, :]`; the output shape
// concatenates along the named axis.
void concat(const float* const* xs, const int64_t* const* shape_in,
            float* y, const int64_t* shape_out, int rank, int n, int axis);

// Inverse of `concat`: split along `axis` into `n` outputs whose sizes are
// given by `axis_sizes[i]`.
void split(const float* x, const int64_t* shape_in,
           float* const* ys, const int64_t* axis_sizes,
           int rank, int n, int axis);

// Gather rows: y[i, :] = x[idx[i], :], where x has shape [vocab, dim].
// Out-of-range indices produce a zero row (matching `embedding`).
void gather_rows(const float* x, const int32_t* idx, float* y,
                 int n_idx, int vocab, int dim);

// Scatter rows: y[idx[i], :] += x[i, :]. y has shape [vocab, dim] and is
// expected to be zero-initialised by the caller. Duplicate indices
// accumulate. Out-of-range indices are dropped.
void scatter_add_rows(const float* x, const int32_t* idx, float* y,
                      int n_idx, int vocab, int dim);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_TRANSFORM_HPP
