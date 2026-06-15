// LonghornAI — benchmark runner.
//
// Each benchmark is a `(name, fn)` pair where `fn` runs the operation under
// test once. The harness handles warm-up, steady-state measurement, and
// percentile reporting. FLOP / byte counts are passed alongside so the
// reporter can produce GFLOPS and GB/s.
//
// All clocks are `steady_clock::now()`; results are written to stdout and,
// optionally, to a CSV file for diffing across runs.
#ifndef LONGHORNAI_BENCH_BENCH_UTIL_HPP
#define LONGHORNAI_BENCH_BENCH_UTIL_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace lh_bench {

struct BenchSpec {
    std::string name;
    std::function<void()> fn;
    double flops = 0.0;     // per-iteration
    double bytes = 0.0;     // per-iteration
    int warmup = 3;
    int iters = 20;
};

struct BenchResult {
    std::string name;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double mean_us = 0.0;
    double gflops = 0.0;
    double gbps = 0.0;
    double flops = 0.0;     // total per-iter (carried through for roofline)
    double bytes = 0.0;
};

// Phase 8 roofline classification. The "ridge point" is the arithmetic
// intensity at which a balanced machine's compute and memory roofs cross
// (peak_flops / peak_bw). Kernels above the ridge are compute-bound; below,
// memory-bound. We treat the band [0.5x, 2x] of the ridge as "balanced" so
// borderline kernels don't flip-flop with measurement noise.
struct RooflineClass {
    enum class Bound { Memory, Balanced, Compute };
    double ai = 0.0;     // FLOPs / byte
    Bound bound = Bound::Balanced;
};

inline const char* bound_str(RooflineClass::Bound b) {
    switch (b) {
        case RooflineClass::Bound::Memory: return "memory";
        case RooflineClass::Bound::Balanced: return "balanced";
        case RooflineClass::Bound::Compute: return "compute";
    }
    return "?";
}

// Default ridge ~10 FLOPs/byte: representative of a contemporary x86 core
// with ~50 GFLOPS/lane and ~5 GB/s sustained per-thread DRAM bandwidth.
// Override for a target machine via the env / CLI.
inline RooflineClass classify_roofline(double flops, double bytes,
                                       double ridge = 10.0) {
    RooflineClass r;
    if (bytes <= 0.0) {
        r.ai = std::numeric_limits<double>::infinity();
        r.bound = RooflineClass::Bound::Compute;
        return r;
    }
    r.ai = flops / bytes;
    if (r.ai < 0.5 * ridge) r.bound = RooflineClass::Bound::Memory;
    else if (r.ai > 2.0 * ridge) r.bound = RooflineClass::Bound::Compute;
    else r.bound = RooflineClass::Bound::Balanced;
    return r;
}

inline double percentile(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double f = pos - lo;
    return v[lo] * (1.0 - f) + v[hi] * f;
}

inline BenchResult run_one(const BenchSpec& spec) {
    for (int i = 0; i < spec.warmup; ++i) spec.fn();
    std::vector<double> us;
    us.reserve(spec.iters);
    for (int i = 0; i < spec.iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        spec.fn();
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    BenchResult r;
    r.name = spec.name;
    r.min_us = *std::min_element(us.begin(), us.end());
    r.max_us = *std::max_element(us.begin(), us.end());
    double sum = 0;
    for (double x : us) sum += x;
    r.mean_us = sum / us.size();
    auto sorted = us;  // percentile() sorts in place; keep a copy.
    r.p50_us = percentile(sorted, 0.50);
    sorted = us;
    r.p90_us = percentile(sorted, 0.90);
    sorted = us;
    r.p99_us = percentile(sorted, 0.99);
    const double secs = r.p50_us * 1.0e-6;
    r.gflops = secs > 0.0 ? (spec.flops / secs) / 1.0e9 : 0.0;
    r.gbps = secs > 0.0 ? (spec.bytes / secs) / 1.0e9 : 0.0;
    r.flops = spec.flops;
    r.bytes = spec.bytes;
    return r;
}

inline void print_header(std::ostream& os) {
    os << std::left << std::setw(40) << "kernel" << std::right << std::setw(12)
       << "p50_us" << std::setw(12) << "p90_us" << std::setw(12) << "p99_us"
       << std::setw(12) << "GFLOP/s" << std::setw(12) << "GB/s" << "\n";
    os << std::string(40 + 12 * 5, '-') << "\n";
}

inline void print_row(std::ostream& os, const BenchResult& r) {
    os << std::left << std::setw(40) << r.name << std::right << std::fixed
       << std::setprecision(2) << std::setw(12) << r.p50_us << std::setw(12)
       << r.p90_us << std::setw(12) << r.p99_us << std::setw(12) << r.gflops
       << std::setw(12) << r.gbps << "\n";
}

inline bool write_csv(const std::string& path,
                      const std::vector<BenchResult>& results) {
    std::ofstream f(path);
    if (!f.good()) return false;
    f << "name,p50_us,p90_us,p99_us,min_us,max_us,mean_us,gflops,gbps\n";
    for (const auto& r : results) {
        f << r.name << "," << r.p50_us << "," << r.p90_us << "," << r.p99_us
          << "," << r.min_us << "," << r.max_us << "," << r.mean_us << ","
          << r.gflops << "," << r.gbps << "\n";
    }
    return true;
}

// Phase 8 roofline output. Per-kernel arithmetic intensity, achieved
// throughput, and bound classification. The CSV is the input to the
// silicon dossier generator and to roofline plots.
inline bool write_roofline_csv(const std::string& path,
                               const std::vector<BenchResult>& results,
                               double ridge = 10.0) {
    std::ofstream f(path);
    if (!f.good()) return false;
    f << "name,ai_flops_per_byte,gflops,gbps,p50_us,bound\n";
    for (const auto& r : results) {
        const auto cls = classify_roofline(r.flops, r.bytes, ridge);
        f << r.name << "," << cls.ai << "," << r.gflops << "," << r.gbps
          << "," << r.p50_us << "," << bound_str(cls.bound) << "\n";
    }
    return true;
}

inline void print_roofline(std::ostream& os,
                           const std::vector<BenchResult>& results,
                           double ridge = 10.0) {
    os << std::left << std::setw(40) << "kernel" << std::right << std::setw(14)
       << "AI(F/B)" << std::setw(12) << "GFLOP/s" << std::setw(12) << "GB/s"
       << std::setw(12) << "bound" << "\n";
    os << std::string(40 + 14 + 12 * 3, '-') << "\n";
    for (const auto& r : results) {
        const auto cls = classify_roofline(r.flops, r.bytes, ridge);
        os << std::left << std::setw(40) << r.name << std::right
           << std::fixed << std::setprecision(2) << std::setw(14) << cls.ai
           << std::setw(12) << r.gflops << std::setw(12) << r.gbps
           << std::setw(12) << bound_str(cls.bound) << "\n";
    }
}

class BenchSuite {
public:
    void add(BenchSpec spec) { specs_.push_back(std::move(spec)); }

    std::vector<BenchResult> run(std::ostream& os = std::cout) {
        std::vector<BenchResult> out;
        out.reserve(specs_.size());
        print_header(os);
        for (const auto& s : specs_) {
            auto r = run_one(s);
            print_row(os, r);
            out.push_back(std::move(r));
        }
        return out;
    }

private:
    std::vector<BenchSpec> specs_;
};

}  // namespace lh_bench

#endif  // LONGHORNAI_BENCH_BENCH_UTIL_HPP
