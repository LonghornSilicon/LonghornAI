// LonghornAI — kernel profiler.
//
// A process-wide registry of (name -> aggregate timing/traffic) records.
// `KernelScope` is the entry point: a RAII timer that, on destruction,
// records elapsed wall-clock plus the FLOP / byte counts the caller passed
// in. The bench harness and future model runner use this to produce
// per-kernel cost tables and Chrome-trace JSON.
//
// Profiling is global and thread-unsafe by design: enable it only for
// single-threaded measurement runs. OpenMP-parallel kernel calls should
// disable profiling or wrap the *outer* call site, not the inner threads.
#ifndef LONGHORNAI_CORE_PROFILER_HPP
#define LONGHORNAI_CORE_PROFILER_HPP

#include <chrono>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace lh {

struct KernelStat {
    std::string name;
    uint64_t calls = 0;
    uint64_t total_ns = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    double total_flops = 0.0;
    double total_bytes = 0.0;
};

class Profiler {
public:
    static Profiler& instance() {
        static Profiler p;
        return p;
    }

    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    void record(const std::string& name, uint64_t ns, double flops,
                double bytes) {
        if (!enabled_) return;
        auto& s = stats_[name];
        if (s.calls == 0) {
            s.name = name;
            s.min_ns = ns;
            s.max_ns = ns;
        } else {
            if (ns < s.min_ns) s.min_ns = ns;
            if (ns > s.max_ns) s.max_ns = ns;
        }
        s.calls += 1;
        s.total_ns += ns;
        s.total_flops += flops;
        s.total_bytes += bytes;
    }

    void reset() { stats_.clear(); }

    std::vector<KernelStat> snapshot() const {
        std::vector<KernelStat> out;
        out.reserve(stats_.size());
        for (const auto& kv : stats_) out.push_back(kv.second);
        return out;
    }

    void print_summary(std::ostream& os) const;

private:
    Profiler() = default;
    bool enabled_ = true;
    std::map<std::string, KernelStat> stats_;
};

class KernelScope {
public:
    KernelScope(const char* name, double flops = 0.0, double bytes = 0.0)
        : name_(name),
          flops_(flops),
          bytes_(bytes),
          start_(std::chrono::steady_clock::now()) {}

    ~KernelScope() {
        const auto end = std::chrono::steady_clock::now();
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
                .count();
        Profiler::instance().record(name_, static_cast<uint64_t>(ns), flops_,
                                    bytes_);
    }

    KernelScope(const KernelScope&) = delete;
    KernelScope& operator=(const KernelScope&) = delete;

private:
    std::string name_;
    double flops_;
    double bytes_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace lh

#endif  // LONGHORNAI_CORE_PROFILER_HPP
