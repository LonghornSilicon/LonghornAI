// LonghornAI — portable low-precision helper types.
//
// Pure C++17 software implementations of IEEE-754 binary16 (`half`) and
// `bfloat16`. No hardware/intrinsic dependency, so the same bits are produced
// on every platform and compiler. The kernel library computes in FP32; these
// types exist for storage and for exercising the FP16/BF16 numeric paths.
#ifndef LONGHORNAI_KERNELS_DTYPES_HPP
#define LONGHORNAI_KERNELS_DTYPES_HPP

#include <cstdint>
#include <cstring>

namespace lh {

namespace detail {

inline uint16_t float_to_half_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t raw_exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x007fffffu;

    if (raw_exp == 0xff) {  // Inf or NaN
        if (mant) return static_cast<uint16_t>(sign | 0x7e00u);  // quiet NaN
        return static_cast<uint16_t>(sign | 0x7c00u);            // Inf
    }

    int32_t exp = static_cast<int32_t>(raw_exp) - 127 + 15;

    if (exp >= 0x1f) {  // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (exp <= 0) {  // subnormal half or underflow to zero
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;  // restore implicit leading 1
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) ++half_mant;
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint16_t result = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    const uint32_t rem = mant & 0x1fffu;
    const uint32_t halfway = 0x1000u;
    // Round to nearest, ties to even. A mantissa carry propagates into the
    // exponent field naturally because the fields are contiguous.
    if (rem > halfway || (rem == halfway && (result & 1u))) ++result;
    return result;
}

inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // signed zero
        } else {       // subnormal half -> normalize into a float
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {  // Inf or NaN
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp - 15 + 127;
        f = sign | (exp << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint16_t float_to_bf16_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    if (((x >> 23) & 0xffu) == 0xffu && (x & 0x007fffffu)) {
        return static_cast<uint16_t>((x >> 16) | 0x0040u);  // keep NaN
    }
    // Round to nearest, ties to even.
    const uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return static_cast<uint16_t>(x >> 16);
}

inline float bf16_bits_to_float(uint16_t b) {
    const uint32_t x = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &x, sizeof(out));
    return out;
}

}  // namespace detail

// IEEE-754 binary16.
struct half {
    uint16_t bits = 0;

    half() = default;
    half(float v) : bits(detail::float_to_half_bits(v)) {}  // NOLINT(runtime/explicit)
    float to_float() const { return detail::half_bits_to_float(bits); }
    operator float() const { return to_float(); }

    static half from_bits(uint16_t b) {
        half h;
        h.bits = b;
        return h;
    }
};

// Brain floating point: the top 16 bits of an FP32 value.
struct bfloat16 {
    uint16_t bits = 0;

    bfloat16() = default;
    bfloat16(float v) : bits(detail::float_to_bf16_bits(v)) {}  // NOLINT(runtime/explicit)
    float to_float() const { return detail::bf16_bits_to_float(bits); }
    operator float() const { return to_float(); }

    static bfloat16 from_bits(uint16_t b) {
        bfloat16 v;
        v.bits = b;
        return v;
    }
};

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_DTYPES_HPP
