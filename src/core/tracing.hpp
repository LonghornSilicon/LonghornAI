// LonghornAI — Chrome trace event emitter.
//
// Companion to `Profiler`. While the profiler aggregates per-kernel
// statistics, the tracer records a flat list of (name, ts, dur) events that
// can be loaded by chrome://tracing or Perfetto for visual inspection of a
// forward pass.
//
// Like the profiler, this is single-threaded by intent; the trace is
// rooted at the first call to `instance()`.
#ifndef LONGHORNAI_CORE_TRACING_HPP
#define LONGHORNAI_CORE_TRACING_HPP

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace lh {

struct TraceEvent {
    std::string name;
    uint64_t ts_us = 0;
    uint64_t dur_us = 0;
};

class Tracer {
public:
    static Tracer& instance() {
        static Tracer t;
        return t;
    }

    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    uint64_t now_us() const {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                now - origin_)
                .count());
    }

    void emit(const std::string& name, uint64_t ts_us, uint64_t dur_us) {
        if (!enabled_) return;
        events_.push_back({name, ts_us, dur_us});
    }

    void reset() { events_.clear(); }

    bool write_chrome_trace(const std::string& path) const {
        std::ofstream f(path);
        if (!f.good()) return false;
        f << "[\n";
        for (size_t i = 0; i < events_.size(); ++i) {
            const auto& e = events_[i];
            f << "  {\"name\":\"" << e.name
              << "\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":" << e.ts_us
              << ",\"dur\":" << e.dur_us << "}";
            if (i + 1 < events_.size()) f << ",";
            f << "\n";
        }
        f << "]\n";
        return true;
    }

    const std::vector<TraceEvent>& events() const { return events_; }

private:
    Tracer() : origin_(std::chrono::steady_clock::now()) {}
    bool enabled_ = false;
    std::chrono::steady_clock::time_point origin_;
    std::vector<TraceEvent> events_;
};

class TraceScope {
public:
    explicit TraceScope(const char* name)
        : name_(name), start_us_(Tracer::instance().now_us()) {}
    ~TraceScope() {
        const uint64_t end_us = Tracer::instance().now_us();
        Tracer::instance().emit(name_, start_us_, end_us - start_us_);
    }

    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;

private:
    std::string name_;
    uint64_t start_us_;
};

}  // namespace lh

#endif  // LONGHORNAI_CORE_TRACING_HPP
