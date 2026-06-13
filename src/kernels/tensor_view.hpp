// LonghornAI — lightweight non-owning tensor view.
//
// A tiny row-major view over an externally-owned buffer. Kernels take raw
// pointers plus dimensions; `TensorView` is a convenience for tests and
// callers that want shape/stride bookkeeping and bounds-checked indexing.
#ifndef LONGHORNAI_KERNELS_TENSOR_VIEW_HPP
#define LONGHORNAI_KERNELS_TENSOR_VIEW_HPP

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace lh {

class TensorView {
public:
    TensorView() = default;

    TensorView(float* data, std::vector<int64_t> shape)
        : data_(data), shape_(std::move(shape)) {
        compute_row_major_strides();
    }

    TensorView(float* data, std::initializer_list<int64_t> shape)
        : data_(data), shape_(shape) {
        compute_row_major_strides();
    }

    float* data() { return data_; }
    const float* data() const { return data_; }

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }

    int64_t rank() const { return static_cast<int64_t>(shape_.size()); }

    int64_t dim(int64_t i) const {
        assert(i >= 0 && i < rank());
        return shape_[static_cast<size_t>(i)];
    }

    int64_t numel() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), int64_t{1},
                               std::multiplies<int64_t>());
    }

    // Row-major linear offset for a full index.
    int64_t offset(std::initializer_list<int64_t> idx) const {
        assert(static_cast<int64_t>(idx.size()) == rank());
        int64_t off = 0;
        int64_t axis = 0;
        for (int64_t i : idx) {
            assert(i >= 0 && i < shape_[static_cast<size_t>(axis)]);
            off += i * strides_[static_cast<size_t>(axis)];
            ++axis;
        }
        return off;
    }

    float& at(std::initializer_list<int64_t> idx) { return data_[offset(idx)]; }
    float at(std::initializer_list<int64_t> idx) const { return data_[offset(idx)]; }

private:
    void compute_row_major_strides() {
        strides_.assign(shape_.size(), 1);
        for (int64_t i = rank() - 2; i >= 0; --i) {
            strides_[static_cast<size_t>(i)] =
                strides_[static_cast<size_t>(i + 1)] *
                shape_[static_cast<size_t>(i + 1)];
        }
    }

    float* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
};

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_TENSOR_VIEW_HPP
