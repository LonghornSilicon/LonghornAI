#include "core/profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <vector>

namespace lh {

void Profiler::print_summary(std::ostream& os) const {
    std::vector<KernelStat> rows;
    rows.reserve(stats_.size());
    for (const auto& kv : stats_) rows.push_back(kv.second);
    std::sort(rows.begin(), rows.end(),
              [](const KernelStat& a, const KernelStat& b) {
                  return a.total_ns > b.total_ns;
              });

    os << std::left << std::setw(28) << "kernel" << std::right << std::setw(10)
       << "calls" << std::setw(14) << "total_ms" << std::setw(14) << "avg_us"
       << std::setw(14) << "GFLOP/s" << std::setw(14) << "GB/s" << "\n";
    os << std::string(28 + 10 + 14 * 4, '-') << "\n";

    for (const auto& s : rows) {
        const double total_ms = s.total_ns / 1.0e6;
        const double avg_us =
            (s.calls ? static_cast<double>(s.total_ns) / s.calls : 0.0) / 1.0e3;
        const double secs = s.total_ns / 1.0e9;
        const double gflops = secs > 0.0 ? (s.total_flops / secs) / 1.0e9 : 0.0;
        const double gbs = secs > 0.0 ? (s.total_bytes / secs) / 1.0e9 : 0.0;
        os << std::left << std::setw(28) << s.name << std::right
           << std::setw(10) << s.calls << std::setw(14) << std::fixed
           << std::setprecision(3) << total_ms << std::setw(14) << avg_us
           << std::setw(14) << gflops << std::setw(14) << gbs << "\n";
    }
}

}  // namespace lh
