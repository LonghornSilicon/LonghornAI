#include "kernels/reduction.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

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

void argmax(const float* x, int32_t* out_idx, float* out_val,
            int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        float m = -std::numeric_limits<float>::infinity();
        int32_t idx = 0;
        for (int i = 0; i < dim; ++i) {
            if (xr[i] > m) {  // strict-greater preserves lowest-index ties
                m = xr[i];
                idx = i;
            }
        }
        if (out_idx) out_idx[r] = idx;
        if (out_val) out_val[r] = m;
    }
}

// Per-row top-k. We keep a length-k min-heap of (value, index) pairs (the
// tiniest sits at heap top), then sort the survivors descending. This is
// O(dim log k) per row, which is the right shape when k << dim.
void topk(const float* x, float* out_val, int32_t* out_idx,
          int rows, int dim, int k) {
    if (k <= 0 || dim <= 0) return;
    if (k > dim) k = dim;

    using Pair = std::pair<float, int32_t>;
    // Min-heap on value: greater-than comparator makes std::pop_heap drop
    // the smallest element to the back.
    auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    std::vector<Pair> heap;
    heap.reserve(static_cast<size_t>(k));

    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * dim;
        heap.clear();
        for (int i = 0; i < dim; ++i) {
            const Pair p{xr[i], i};
            if (static_cast<int>(heap.size()) < k) {
                heap.push_back(p);
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else if (xr[i] > heap.front().first ||
                       (xr[i] == heap.front().first &&
                        i < heap.front().second)) {
                // Beats the current minimum, or ties and is earlier-indexed
                // (tie-break by lowest index for determinism).
                std::pop_heap(heap.begin(), heap.end(), cmp);
                heap.back() = p;
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
        // Sort survivors descending by value, ties broken by ascending index.
        std::sort(heap.begin(), heap.end(),
                  [](const Pair& a, const Pair& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        const int64_t base = static_cast<int64_t>(r) * k;
        for (int i = 0; i < k; ++i) {
            if (out_val) out_val[base + i] = heap[static_cast<size_t>(i)].first;
            if (out_idx) out_idx[base + i] = heap[static_cast<size_t>(i)].second;
        }
    }
}

}  // namespace lh
