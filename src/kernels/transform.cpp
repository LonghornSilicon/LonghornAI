#include "kernels/transform.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace lh {

namespace {

// Row-major strides for a shape; strides[i] is the count of elements per
// step along axis i.
inline std::vector<int64_t> row_major_strides(const int64_t* shape, int rank) {
    std::vector<int64_t> s(static_cast<size_t>(rank), 1);
    for (int i = rank - 2; i >= 0; --i) {
        s[static_cast<size_t>(i)] =
            s[static_cast<size_t>(i + 1)] *
            shape[static_cast<size_t>(i + 1)];
    }
    return s;
}

inline int64_t numel(const int64_t* shape, int rank) {
    int64_t n = 1;
    for (int i = 0; i < rank; ++i) n *= shape[i];
    return n;
}

}  // namespace

void transpose2d(const float* x, float* y, int M, int N) {
    // 32-element row tiles keep the destination writes contiguous in chunks
    // and the source reads fall on consecutive cache lines per tile.
    constexpr int TM = 32;
    constexpr int TN = 32;
    for (int i0 = 0; i0 < M; i0 += TM) {
        const int imax = std::min(M, i0 + TM);
        for (int j0 = 0; j0 < N; j0 += TN) {
            const int jmax = std::min(N, j0 + TN);
            for (int i = i0; i < imax; ++i) {
                for (int j = j0; j < jmax; ++j) {
                    y[static_cast<int64_t>(j) * M + i] =
                        x[static_cast<int64_t>(i) * N + j];
                }
            }
        }
    }
}

void permute(const float* x, float* y, const int64_t* shape,
             const int* perm, int rank) {
    if (rank <= 0) return;
    const auto in_strides = row_major_strides(shape, rank);

    // Output shape = shape[perm[i]]
    std::vector<int64_t> out_shape(static_cast<size_t>(rank));
    for (int i = 0; i < rank; ++i) {
        out_shape[static_cast<size_t>(i)] =
            shape[static_cast<size_t>(perm[i])];
    }
    const auto out_strides = row_major_strides(out_shape.data(), rank);

    // Walk the output in linear order; for each linear position, decompose
    // into multi-index using out_strides, then map back through `perm` to
    // get the source index. This is the obvious-correct reference; the
    // optimised path tile-iterates the contiguous trailing dim.
    const int64_t total = numel(shape, rank);
    std::vector<int64_t> out_idx(static_cast<size_t>(rank), 0);
    for (int64_t lin = 0; lin < total; ++lin) {
        int64_t rem = lin;
        for (int i = 0; i < rank; ++i) {
            out_idx[static_cast<size_t>(i)] =
                rem / out_strides[static_cast<size_t>(i)];
            rem %= out_strides[static_cast<size_t>(i)];
        }
        int64_t src = 0;
        for (int i = 0; i < rank; ++i) {
            src += out_idx[static_cast<size_t>(i)] *
                   in_strides[static_cast<size_t>(perm[i])];
        }
        y[lin] = x[src];
    }
}

void concat(const float* const* xs, const int64_t* const* shape_in,
            float* y, const int64_t* shape_out, int rank, int n, int axis) {
    if (rank <= 0 || n <= 0) return;

    // Outer = product of dims [0, axis), Inner = product of dims (axis, rank).
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape_out[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; ++i) inner *= shape_out[i];

    int64_t axis_offset = 0;  // running output offset along the concat axis
    for (int t = 0; t < n; ++t) {
        const int64_t in_axis = shape_in[t][axis];
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t a = 0; a < in_axis; ++a) {
                const int64_t dst =
                    o * (shape_out[axis] * inner) +
                    (axis_offset + a) * inner;
                const int64_t src = o * (in_axis * inner) + a * inner;
                std::memcpy(y + dst, xs[t] + src,
                            inner * sizeof(float));
            }
        }
        axis_offset += in_axis;
    }
}

void split(const float* x, const int64_t* shape_in,
           float* const* ys, const int64_t* axis_sizes,
           int rank, int n, int axis) {
    if (rank <= 0 || n <= 0) return;
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape_in[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; ++i) inner *= shape_in[i];
    const int64_t in_axis = shape_in[axis];

    int64_t axis_offset = 0;
    for (int t = 0; t < n; ++t) {
        const int64_t k = axis_sizes[t];
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t a = 0; a < k; ++a) {
                const int64_t src =
                    o * (in_axis * inner) + (axis_offset + a) * inner;
                const int64_t dst = o * (k * inner) + a * inner;
                std::memcpy(ys[t] + dst, x + src, inner * sizeof(float));
            }
        }
        axis_offset += k;
    }
}

void gather_rows(const float* x, const int32_t* idx, float* y,
                 int n_idx, int vocab, int dim) {
    for (int i = 0; i < n_idx; ++i) {
        const int32_t id = idx[i];
        float* dst = y + static_cast<int64_t>(i) * dim;
        if (id < 0 || id >= vocab) {
            std::memset(dst, 0, sizeof(float) * static_cast<size_t>(dim));
            continue;
        }
        const float* src = x + static_cast<int64_t>(id) * dim;
        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(dim));
    }
}

void scatter_add_rows(const float* x, const int32_t* idx, float* y,
                      int n_idx, int vocab, int dim) {
    for (int i = 0; i < n_idx; ++i) {
        const int32_t id = idx[i];
        if (id < 0 || id >= vocab) continue;
        const float* src = x + static_cast<int64_t>(i) * dim;
        float* dst = y + static_cast<int64_t>(id) * dim;
        for (int d = 0; d < dim; ++d) dst[d] += src[d];
    }
}

}  // namespace lh
