// LonghornAI — numerical validation utilities.
//
// Tools for measuring how far an optimized kernel drifts from its naive
// reference. Provides ULP comparison for FP32 plus per-dtype tolerance
// policies. Used by the gtest suite (`tests/numerical/`) and by ad-hoc
// validation scripts.
#ifndef LONGHORNAI_TESTS_NUMERICAL_ULP_HPP
#define LONGHORNAI_TESTS_NUMERICAL_ULP_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "core/dtype.hpp"

namespace lh_num {

// ULP distance between two FP32 values, in their integer representation.
// Returns 0 for exact equality (incl. signed zeros) and a large number for
// pairs that span a sign change or include a NaN.
inline uint64_t ulp_distance_f32(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<uint64_t>::max();
    }
    if (a == b) return 0;
    int32_t ai, bi;
    std::memcpy(&ai, &a, sizeof(ai));
    std::memcpy(&bi, &b, sizeof(bi));
    if ((ai < 0) != (bi < 0)) {
        // Signs differ; treat as far apart unless both are zero (handled
        // above by `a == b`).
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t ua = static_cast<uint64_t>(std::abs(ai));
    const uint64_t ub = static_cast<uint64_t>(std::abs(bi));
    return (ua > ub) ? (ua - ub) : (ub - ua);
}

struct ToleranceBounds {
    float atol;
    float rtol;
};

// Defaults derived from the working numerical-validation policy in PLAN.md.
inline ToleranceBounds default_tolerance(lh::DType dt) {
    switch (dt) {
        case lh::DType::F32: return {1e-7f, 1e-5f};
        case lh::DType::F16: return {1e-4f, 1e-3f};
        case lh::DType::BF16: return {5e-4f, 5e-3f};
        default: return {0.0f, 0.0f};  // integer dtypes: exact match
    }
}

struct ErrorReport {
    uint64_t n = 0;
    uint64_t n_violations = 0;     // exceeds tol
    uint64_t max_ulp = 0;
    double mean_ulp = 0.0;
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    double max_rel_err = 0.0;
};

inline ErrorReport sweep(const std::vector<float>& got,
                         const std::vector<float>& ref,
                         ToleranceBounds tol) {
    ErrorReport r;
    if (got.size() != ref.size()) {
        r.n_violations = 1;
        r.max_ulp = std::numeric_limits<uint64_t>::max();
        return r;
    }
    r.n = got.size();
    if (r.n == 0) return r;
    long double sum_abs = 0.0L;
    long double sum_ulp = 0.0L;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = got[i];
        const float b = ref[i];
        const double abs_err = std::fabs(static_cast<double>(a) - b);
        const double rel_err =
            (std::fabs(b) > 0.0) ? abs_err / std::fabs(b) : abs_err;
        const uint64_t u = ulp_distance_f32(a, b);
        sum_abs += abs_err;
        if (u != std::numeric_limits<uint64_t>::max()) sum_ulp += u;
        if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;
        if (rel_err > r.max_rel_err) r.max_rel_err = rel_err;
        if (u != std::numeric_limits<uint64_t>::max() && u > r.max_ulp) {
            r.max_ulp = u;
        }
        const double bound =
            static_cast<double>(tol.atol) + static_cast<double>(tol.rtol) *
                                                std::fabs(static_cast<double>(b));
        if (abs_err > bound || std::isnan(abs_err)) ++r.n_violations;
    }
    r.mean_abs_err = static_cast<double>(sum_abs) / r.n;
    r.mean_ulp = static_cast<double>(sum_ulp) / r.n;
    return r;
}

}  // namespace lh_num

#endif  // LONGHORNAI_TESTS_NUMERICAL_ULP_HPP
