// Shared test helpers: deterministic random tensors and a tolerance-based
// comparator that plugs into GoogleTest assertions.
#ifndef LONGHORNAI_TESTS_TEST_UTIL_HPP
#define LONGHORNAI_TESTS_TEST_UTIL_HPP

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lh_test {

// Default per-dtype tolerances. FP32 kernels are validated tightly; the
// FP16/BF16 round-trip tests use their own looser bounds.
constexpr float kAtolF32 = 1e-4f;
constexpr float kRtolF32 = 1e-4f;

inline std::vector<float> random_vector(int64_t n, uint32_t seed,
                                        float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

inline std::vector<int32_t> random_ids(int64_t n, int32_t vocab,
                                       uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, vocab - 1);
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

// Element-wise closeness using combined absolute + relative tolerance.
inline ::testing::AssertionResult AllClose(const std::vector<float>& a,
                                           const std::vector<float>& b,
                                           float atol = kAtolF32,
                                           float rtol = kRtolF32) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure()
               << "size mismatch: " << a.size() << " vs " << b.size();
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float tol = atol + rtol * std::fabs(b[i]);
        if (diff > tol || std::isnan(diff)) {
            return ::testing::AssertionFailure()
                   << "mismatch at index " << i << ": " << a[i] << " vs "
                   << b[i] << " (|diff|=" << diff << ", tol=" << tol << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace lh_test

#endif  // LONGHORNAI_TESTS_TEST_UTIL_HPP
