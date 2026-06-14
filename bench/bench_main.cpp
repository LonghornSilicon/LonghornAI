// LonghornAI — bench main.
//
// Single executable that registers benchmarks for every kernel family and
// drives them through `BenchSuite`. Shape suites are inlined here for now;
// once Phase 0 scaffolding settles, they migrate to YAML files under
// `bench/shapes/`.
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bench_util.hpp"
#include "kernels/activation.hpp"
#include "kernels/attention.hpp"
#include "kernels/embedding.hpp"
#include "kernels/gemm.hpp"
#include "kernels/normalization.hpp"
#include "kernels/reduction.hpp"
#include "kernels/rope.hpp"
#include "kernels/softmax.hpp"

namespace {

std::vector<float> rand_f32(int64_t n, uint32_t seed, float lo = -1.0f,
                            float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

// Llama-class shape sketch: (M, N, K) for the four projection GEMMs. Sized
// down so the bench finishes quickly on a developer laptop; absolute numbers
// are illustrative, not validation gates.
struct GemmShape {
    const char* tag;
    int M, N, K;
};

const GemmShape kGemmShapes[] = {
    {"gemm/qkv_small", 128, 768, 768},
    {"gemm/mlp_up", 128, 2048, 768},
    {"gemm/mlp_down", 128, 768, 2048},
};

void register_gemm(lh_bench::BenchSuite& suite) {
    for (const auto& s : kGemmShapes) {
        auto A = rand_f32(int64_t(s.M) * s.K, 1);
        auto B = rand_f32(int64_t(s.K) * s.N, 2);
        std::vector<float> C(static_cast<size_t>(s.M) * s.N, 0.0f);
        lh_bench::BenchSpec spec;
        spec.name = std::string(s.tag) + "[" + std::to_string(s.M) + "x" +
                    std::to_string(s.N) + "x" + std::to_string(s.K) + "]";
        spec.flops = 2.0 * s.M * s.N * s.K;
        spec.bytes = 4.0 * (int64_t(s.M) * s.K + int64_t(s.K) * s.N +
                            int64_t(s.M) * s.N);
        // Capture by reference where possible; A/B/C outlive the bench run
        // because they're stack-allocated in this function and the suite is
        // executed before we return.
        spec.fn = [A, B, C, s]() mutable {
            lh::gemm(A.data(), B.data(), C.data(), s.M, s.N, s.K);
        };
        suite.add(std::move(spec));
    }
}

void register_norm_softmax(lh_bench::BenchSuite& suite) {
    const int rows = 128;
    const int dim = 4096;
    auto x = rand_f32(int64_t(rows) * dim, 10);
    auto g = rand_f32(dim, 11, 0.5f, 1.5f);
    std::vector<float> y(x.size(), 0.0f);

    {
        lh_bench::BenchSpec s;
        s.name = "rmsnorm[128x4096]";
        s.bytes = 2.0 * rows * dim * 4.0;
        s.fn = [x, g, y]() mutable {
            lh::rmsnorm(x.data(), g.data(), y.data(), rows, dim);
        };
        suite.add(std::move(s));
    }
    {
        lh_bench::BenchSpec s;
        s.name = "softmax[128x4096]";
        s.bytes = 2.0 * rows * dim * 4.0;
        s.fn = [x, y]() mutable {
            lh::softmax(x.data(), y.data(), rows, dim);
        };
        suite.add(std::move(s));
    }
}

void register_attention(lh_bench::BenchSuite& suite) {
    lh::AttnConfig cfg;
    cfg.batch = 1;
    cfg.n_q_heads = 8;
    cfg.n_kv_heads = 8;
    cfg.seq_q = 128;
    cfg.seq_k = 128;
    cfg.head_dim = 64;
    cfg.causal = true;

    const int64_t qn = int64_t(cfg.batch) * cfg.n_q_heads * cfg.seq_q * cfg.head_dim;
    const int64_t kn = int64_t(cfg.batch) * cfg.n_kv_heads * cfg.seq_k * cfg.head_dim;
    auto Q = rand_f32(qn, 100);
    auto K = rand_f32(kn, 101);
    auto V = rand_f32(kn, 102);
    std::vector<float> O(static_cast<size_t>(qn), 0.0f);

    {
        lh_bench::BenchSpec s;
        s.name = "sdpa/causal[1x8x128x64]";
        s.fn = [Q, K, V, O, cfg]() mutable {
            lh::sdpa(Q.data(), K.data(), V.data(), O.data(), cfg);
        };
        suite.add(std::move(s));
    }
    {
        lh_bench::BenchSpec s;
        s.name = "flash/causal[1x8x128x64]";
        s.fn = [Q, K, V, O, cfg]() mutable {
            lh::flash_attention(Q.data(), K.data(), V.data(), O.data(), cfg, 32);
        };
        suite.add(std::move(s));
    }
}

}  // namespace

int main(int argc, char** argv) {
    lh_bench::BenchSuite suite;
    register_gemm(suite);
    register_norm_softmax(suite);
    register_attention(suite);

    auto results = suite.run(std::cout);

    std::string csv_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--csv") csv_path = argv[i + 1];
    }
    if (!csv_path.empty()) {
        if (lh_bench::write_csv(csv_path, results)) {
            std::cout << "wrote " << csv_path << "\n";
        } else {
            std::cerr << "failed to write " << csv_path << "\n";
        }
    }
    return 0;
}
