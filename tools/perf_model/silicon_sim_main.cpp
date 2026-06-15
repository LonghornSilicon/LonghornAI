// LonghornAI — Stage A simulator entry point.
//
// Runs the default kernel suite through the analytical performance model
// for both edge and server profiles, prints a per-kernel prediction
// table, and optionally writes a CSV.
#include <iomanip>
#include <iostream>
#include <string>

#include "perf_model.hpp"

namespace {

void print_header(std::ostream& os, const std::string& title) {
    os << "\n=== " << title << " ===\n";
    os << std::left << std::setw(40) << "kernel" << std::setw(20) << "engine"
       << std::right << std::setw(10) << "AI" << std::setw(14)
       << "t_compute_us" << std::setw(14) << "t_memory_us" << std::setw(14)
       << "t_us" << std::setw(12) << "bound" << "\n";
    os << std::string(40 + 20 + 10 + 14 * 3 + 12, '-') << "\n";
}

void print_row(std::ostream& os, const lh_perf::Prediction& p) {
    os << std::left << std::setw(40) << p.name << std::setw(20)
       << lh_perf::engine_name(p.engine) << std::right << std::fixed
       << std::setprecision(2) << std::setw(10) << p.ai_flops_per_byte
       << std::setw(14) << p.t_compute_s * 1.0e6 << std::setw(14)
       << p.t_memory_s * 1.0e6 << std::setw(14) << p.t_predicted_s * 1.0e6
       << std::setw(12) << lh_perf::bound_name(p.bound) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    bool server = false;
    std::string csv_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--server") server = true;
        else if (a == "--edge") server = false;
        else if (a == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    const auto cfg = server ? lh_perf::LonghornConfig::server()
                            : lh_perf::LonghornConfig::edge();
    const auto kernels = lh_perf::default_bench_suite();
    const auto preds = lh_perf::predict_all(kernels, cfg);

    print_header(std::cout,
                 server ? "Longhorn server profile" : "Longhorn edge profile");
    for (const auto& p : preds) print_row(std::cout, p);

    if (!csv_path.empty()) {
        if (lh_perf::write_predictions_csv(csv_path, preds)) {
            std::cout << "wrote " << csv_path << "\n";
        } else {
            std::cerr << "failed to write " << csv_path << "\n";
        }
    }
    return 0;
}
