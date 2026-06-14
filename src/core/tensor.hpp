// LonghornAI — owning Tensor.
//
// Holds a raw byte buffer + dtype + shape, allocated through an `IAllocator`.
// Kernels still operate on raw pointers; this exists so callers (tests, bench
// harness, future model runner) can manage tensor lifetime without cluttering
// kernel signatures.
//
// Move-only. Row-major. Strides are derived from shape on construction; if
// you need a non-contiguous view, take a `TensorView` over the data instead.
#ifndef LONGHORNAI_CORE_TENSOR_HPP
#define LONGHORNAI_CORE_TENSOR_HPP

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "core/allocator.hpp"
#include "core/dtype.hpp"

namespace lh {

class Tensor {
public:
    Tensor() = default;

    Tensor(DType dt, std::vector<int64_t> shape,
           IAllocator& alloc = HeapAllocator::instance())
        : alloc_(&alloc), dtype_(dt), shape_(std::move(shape)) {
        compute_row_major_strides();
        const size_t bytes = byte_size();
        data_ = bytes ? alloc_->alloc(bytes, alignof(double)) : nullptr;
    }

    ~Tensor() { release(); }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept { steal(std::move(other)); }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            steal(std::move(other));
        }
        return *this;
    }

    DType dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }

    int64_t rank() const { return static_cast<int64_t>(shape_.size()); }

    int64_t numel() const {
        if (shape_.empty()) return 0;
        int64_t n = 1;
        for (auto d : shape_) n *= d;
        return n;
    }

    size_t byte_size() const {
        return static_cast<size_t>(numel()) * dtype_size(dtype_);
    }

    void* raw() { return data_; }
    const void* raw() const { return data_; }

    template <class T> T* data() { return static_cast<T*>(data_); }
    template <class T> const T* data() const { return static_cast<const T*>(data_); }

    void zero() {
        if (data_) std::memset(data_, 0, byte_size());
    }

    // Factory: an FP32 tensor pre-zeroed (handy for kernel outputs).
    static Tensor zeros_f32(std::vector<int64_t> shape,
                            IAllocator& alloc = HeapAllocator::instance()) {
        Tensor t(DType::F32, std::move(shape), alloc);
        t.zero();
        return t;
    }

private:
    void compute_row_major_strides() {
        strides_.assign(shape_.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape_.size()) - 2; i >= 0; --i) {
            strides_[static_cast<size_t>(i)] =
                strides_[static_cast<size_t>(i + 1)] *
                shape_[static_cast<size_t>(i + 1)];
        }
    }

    void release() {
        if (data_ && alloc_) alloc_->free(data_);
        data_ = nullptr;
    }

    void steal(Tensor&& other) {
        alloc_ = other.alloc_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        data_ = other.data_;
        other.data_ = nullptr;
        other.alloc_ = nullptr;
    }

    IAllocator* alloc_ = nullptr;
    DType dtype_ = DType::F32;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    void* data_ = nullptr;
};

}  // namespace lh

#endif  // LONGHORNAI_CORE_TENSOR_HPP
