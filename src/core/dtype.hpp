// LonghornAI — runtime dtype tag.
//
// A small enum + size helper used by `Tensor` and the bench/profiling
// harness. The kernel library itself stays templated on raw element types;
// this exists for code that holds tensors in containers and for profiling
// readouts.
#ifndef LONGHORNAI_CORE_DTYPE_HPP
#define LONGHORNAI_CORE_DTYPE_HPP

#include <cstddef>
#include <cstdint>

namespace lh {

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    I8 = 4,
    U8 = 5,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I32: return 4;
        case DType::I8: return 1;
        case DType::U8: return 1;
    }
    return 0;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::I32: return "i32";
        case DType::I8: return "i8";
        case DType::U8: return "u8";
    }
    return "?";
}

}  // namespace lh

#endif  // LONGHORNAI_CORE_DTYPE_HPP
