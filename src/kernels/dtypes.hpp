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

// FP8 E4M3: 1 sign + 4 exponent (bias 7) + 3 mantissa.
// Range: about [-448, 448]. Single NaN encoding (sign-extended 0x7f); no
// infinities (the slot otherwise reserved for inf is recycled as the max
// finite). The H100 / MI300 datapath dtype for weights and activations.
inline uint8_t float_to_e4m3_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 24) & 0x80u;
    const uint32_t raw_exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x007fffffu;

    if (raw_exp == 0xff) {  // NaN or +/-inf in fp32 → NaN in E4M3
        return static_cast<uint8_t>(sign | 0x7fu);
    }

    const int unbiased = static_cast<int>(raw_exp) - 127;
    int new_exp = unbiased + 7;  // E4M3 bias = 7

    // Saturate to max-finite (0x7e for +; sign bit OR'd separately).
    // Max finite encoding in E4M3 is S.1111.110 = 1.75 * 2^8 = 448.
    if (new_exp > 15 || (new_exp == 15 && mant > 0x600000u)) {
        return static_cast<uint8_t>(sign | 0x7eu);
    }

    if (new_exp >= 1) {
        // Normal: top 3 bits of mantissa, round-to-nearest-even on the
        // discarded bits.
        const uint32_t result_mant = mant >> 20;
        const uint32_t rem = mant & 0xfffffu;
        const uint32_t halfway = 0x80000u;
        uint32_t out = (static_cast<uint32_t>(new_exp) << 3) | result_mant;
        if (rem > halfway || (rem == halfway && (out & 1u))) {
            ++out;
            if (out >= 0x80u) {  // mantissa carry → exponent bumps
                if ((out >> 3) > 15) return static_cast<uint8_t>(sign | 0x7eu);
            }
        }
        return static_cast<uint8_t>(sign | (out & 0x7fu));
    }

    // Subnormal in E4M3: new_exp <= 0. Shift mantissa right by (1-new_exp+20)
    // so the implicit leading 1 lands in the right bit position, then round.
    if (new_exp < -2) return static_cast<uint8_t>(sign);  // underflow to 0
    mant |= 0x00800000u;  // restore implicit leading 1
    const int shift = 1 - new_exp + 20;
    const uint32_t result_mant = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    uint32_t out = result_mant;
    if (rem > halfway || (rem == halfway && (out & 1u))) ++out;
    return static_cast<uint8_t>(sign | (out & 0x7fu));
}

inline float e4m3_bits_to_float(uint8_t b) {
    const uint32_t sign = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t exp = (b >> 3) & 0x0fu;
    const uint32_t mant = b & 0x07u;

    if (exp == 0x0f && mant == 0x07) {
        // E4M3's NaN encoding (the only non-finite).
        const uint32_t bits = sign | 0x7fc00000u;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // signed zero
        } else {
            // Subnormal: normalize into a float.
            uint32_t m = mant;
            int e = 1;
            while ((m & 0x08u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x07u;
            const uint32_t fexp = static_cast<uint32_t>(127 - 7 + e);
            f = sign | (fexp << 23) | (m << 20);
        }
    } else {
        const uint32_t fexp = exp - 7 + 127;
        f = sign | (fexp << 23) | (mant << 20);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// FP8 E5M2: 1 sign + 5 exponent (bias 15) + 2 mantissa. Wider range
// (~6.5e4) at the cost of precision. Used for activation gradients and
// dynamic-range-sensitive paths; we expose it for completeness.
inline uint8_t float_to_e5m2_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 24) & 0x80u;
    const uint32_t raw_exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x007fffffu;

    if (raw_exp == 0xff) {
        if (mant) return static_cast<uint8_t>(sign | 0x7fu);  // NaN
        return static_cast<uint8_t>(sign | 0x7cu);             // Inf
    }

    const int unbiased = static_cast<int>(raw_exp) - 127;
    int new_exp = unbiased + 15;

    if (new_exp >= 31) return static_cast<uint8_t>(sign | 0x7cu);  // overflow → Inf

    if (new_exp >= 1) {
        const uint32_t result_mant = mant >> 21;
        const uint32_t rem = mant & 0x1fffffu;
        const uint32_t halfway = 0x100000u;
        uint32_t out = (static_cast<uint32_t>(new_exp) << 2) | result_mant;
        if (rem > halfway || (rem == halfway && (out & 1u))) ++out;
        if (out >= 0x7cu) return static_cast<uint8_t>(sign | 0x7cu);
        return static_cast<uint8_t>(sign | (out & 0x7fu));
    }

    if (new_exp < -1) return static_cast<uint8_t>(sign);
    mant |= 0x00800000u;
    const int shift = 1 - new_exp + 21;
    const uint32_t result_mant = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    uint32_t out = result_mant;
    if (rem > halfway || (rem == halfway && (out & 1u))) ++out;
    return static_cast<uint8_t>(sign | (out & 0x7fu));
}

inline float e5m2_bits_to_float(uint8_t b) {
    const uint32_t sign = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t exp = (b >> 2) & 0x1fu;
    const uint32_t mant = b & 0x03u;

    uint32_t f;
    if (exp == 0x1f) {
        // Inf / NaN
        f = sign | 0x7f800000u | (mant << 21);
    } else if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t m = mant;
            int e = 1;
            while ((m & 0x04u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x03u;
            const uint32_t fexp = static_cast<uint32_t>(127 - 15 + e);
            f = sign | (fexp << 23) | (m << 21);
        }
    } else {
        const uint32_t fexp = exp - 15 + 127;
        f = sign | (fexp << 23) | (mant << 21);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
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

// 8-bit float, E4M3 variant. Workhorse FP8 dtype for weights/activations.
struct fp8_e4m3 {
    uint8_t bits = 0;

    fp8_e4m3() = default;
    fp8_e4m3(float v) : bits(detail::float_to_e4m3_bits(v)) {}  // NOLINT(runtime/explicit)
    float to_float() const { return detail::e4m3_bits_to_float(bits); }
    operator float() const { return to_float(); }

    static fp8_e4m3 from_bits(uint8_t b) {
        fp8_e4m3 v;
        v.bits = b;
        return v;
    }
};

// 8-bit float, E5M2 variant. Wider exponent range for dynamic-range-
// sensitive paths.
struct fp8_e5m2 {
    uint8_t bits = 0;

    fp8_e5m2() = default;
    fp8_e5m2(float v) : bits(detail::float_to_e5m2_bits(v)) {}  // NOLINT(runtime/explicit)
    float to_float() const { return detail::e5m2_bits_to_float(bits); }
    operator float() const { return to_float(); }

    static fp8_e5m2 from_bits(uint8_t b) {
        fp8_e5m2 v;
        v.bits = b;
        return v;
    }
};

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_DTYPES_HPP
