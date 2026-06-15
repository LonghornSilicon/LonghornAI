#include "perf_model.hpp"

#include <algorithm>
#include <fstream>
#include <limits>

namespace lh_perf {

const char* engine_name(Engine e) {
    switch (e) {
        case Engine::TensorUnit: return "TensorUnit";
        case Engine::AttentionEngine: return "AttentionEngine";
        case Engine::KvController: return "KvController";
        case Engine::VectorUnit: return "VectorUnit";
        case Engine::NormEngine: return "NormEngine";
        case Engine::SoftmaxEngine: return "SoftmaxEngine";
        case Engine::ReductionEngine: return "ReductionEngine";
        case Engine::PermutationEngine: return "PermutationEngine";
        case Engine::ScanEngine: return "ScanEngine";
        case Engine::DmaEngine: return "DmaEngine";
        case Engine::HostScheduler: return "HostScheduler";
    }
    return "?";
}

const char* bound_name(Bound b) {
    switch (b) {
        case Bound::Memory: return "memory";
        case Bound::Balanced: return "balanced";
        case Bound::Compute: return "compute";
    }
    return "?";
}

LonghornConfig LonghornConfig::edge() {
    LonghornConfig c;
    c.tensor_peak_fp16 = 4.0e12;          // 4 TFLOPS
    c.tensor_tiles = 1;
    c.vector_peak = 256.0e9;
    // The small streaming engines (Norm, Softmax, Reduction, Scan) are
    // architecturally bandwidth-bound: they process inputs at L1 rate.
    // Their FLOP peaks are therefore ~the L1 element rate * FLOPs/element.
    // Setting these high enough that "memory-bound dossier" kernels stay
    // memory-bound (the architectural invariant per docs/silicon/roofline.md).
    c.norm_peak = 256.0e9;
    c.softmax_peak = 256.0e9;
    c.reduction_peak = 256.0e9;
    c.scan_peak = 1.0e12;
    c.permute_peak = 256.0e9;
    c.bw_l1_per_tile = 1.0e12;
    c.bw_l2 = 500.0e9;
    c.bw_hbm = 50.0e9;                    // LPDDR5X effective
    c.bw_lpddr = 50.0e9;
    c.host_peak = 1.0e10;
    return c;
}

LonghornConfig LonghornConfig::server() {
    LonghornConfig c;
    c.tensor_peak_fp16 = 16.0e12;         // 16 TFLOPS per tile
    c.tensor_tiles = 8;
    c.vector_peak = 256.0e9;              // per tile
    c.norm_peak = 256.0e9;                // per tile, bandwidth-bound
    c.softmax_peak = 256.0e9;
    c.reduction_peak = 256.0e9;
    c.scan_peak = 1.0e12;
    c.permute_peak = 1.0e12;              // wider on server
    c.bw_l1_per_tile = 1.0e12;
    c.bw_l2 = 500.0e9;
    c.bw_hbm = 2.8e12;                    // 4 stacks HBM3e * 0.7 utilisation
    c.bw_lpddr = 100.0e9;
    c.host_peak = 1.0e10;
    return c;
}

namespace {

double engine_compute_peak(const LonghornConfig& cfg, Engine e,
                           double dtype_mult) {
    switch (e) {
        case Engine::TensorUnit:
            return cfg.tensor_peak_fp16 * dtype_mult * cfg.tensor_tiles;
        case Engine::AttentionEngine:
            // Attention shares Tensor-Unit-equivalent throughput on the
            // QKᵀ and PV matmuls. We charge half a tile because the
            // online-softmax + bias-add lanes share the engine path.
            return 0.5 * cfg.tensor_peak_fp16 * dtype_mult;
        case Engine::KvController:
            // Bandwidth-only block; FLOPs (the dequant rescale) are
            // negligible relative to bytes. Set a high "compute peak" so
            // memory always wins.
            return 1.0e15;
        case Engine::VectorUnit:
            return cfg.vector_peak * cfg.tensor_tiles;
        case Engine::NormEngine:
            return cfg.norm_peak * cfg.tensor_tiles;
        case Engine::SoftmaxEngine:
            return cfg.softmax_peak * cfg.tensor_tiles;
        case Engine::ReductionEngine:
            return cfg.reduction_peak * cfg.tensor_tiles;
        case Engine::ScanEngine:
            return cfg.scan_peak * cfg.tensor_tiles;
        case Engine::PermutationEngine:
            return cfg.permute_peak;
        case Engine::DmaEngine:
            // Pure data motion: compute path doesn't bind. Same trick.
            return 1.0e15;
        case Engine::HostScheduler:
            return cfg.host_peak;
    }
    return 1.0e12;
}

double mem_bandwidth(const LonghornConfig& cfg, MemTier t) {
    switch (t) {
        case MemTier::HBM: return cfg.bw_hbm;
        case MemTier::LPDDR: return cfg.bw_lpddr;
        case MemTier::L2: return cfg.bw_l2;
        case MemTier::L1: return cfg.bw_l1_per_tile * cfg.tensor_tiles;
    }
    return cfg.bw_hbm;
}

}  // namespace

Prediction predict(const KernelDesc& k, const LonghornConfig& cfg) {
    Prediction p;
    p.name = k.name;
    p.engine = k.engine;
    const double peak_compute =
        engine_compute_peak(cfg, k.engine, k.dtype_throughput_mult);
    const double peak_bw = mem_bandwidth(cfg, k.mem_tier);

    p.t_compute_s = (peak_compute > 0.0) ? (k.flops / peak_compute) : 0.0;
    p.t_memory_s = (peak_bw > 0.0) ? (k.bytes / peak_bw) : 0.0;
    p.t_predicted_s = std::max(p.t_compute_s, p.t_memory_s);
    p.ai_flops_per_byte =
        (k.bytes > 0.0) ? (k.flops / k.bytes)
                        : std::numeric_limits<double>::infinity();

    // Bound class: "balanced" if the two roofs are within a factor of
    // 2 — matches the bench harness's classifier shape (a roughly 2×
    // band around the ridge).
    if (p.t_compute_s == 0.0 && p.t_memory_s == 0.0) {
        p.bound = Bound::Balanced;
    } else if (p.t_compute_s > 2.0 * p.t_memory_s) {
        p.bound = Bound::Compute;
    } else if (p.t_memory_s > 2.0 * p.t_compute_s) {
        p.bound = Bound::Memory;
    } else {
        p.bound = Bound::Balanced;
    }
    return p;
}

std::vector<Prediction> predict_all(const std::vector<KernelDesc>& kernels,
                                    const LonghornConfig& cfg) {
    std::vector<Prediction> out;
    out.reserve(kernels.size());
    for (const auto& k : kernels) out.push_back(predict(k, cfg));
    return out;
}

bool write_predictions_csv(const std::string& path,
                           const std::vector<Prediction>& preds) {
    std::ofstream f(path);
    if (!f.good()) return false;
    f << "name,engine,ai,t_compute_us,t_memory_us,t_predicted_us,bound\n";
    for (const auto& p : preds) {
        f << p.name << "," << engine_name(p.engine) << ","
          << p.ai_flops_per_byte << "," << p.t_compute_s * 1.0e6 << ","
          << p.t_memory_s * 1.0e6 << "," << p.t_predicted_s * 1.0e6 << ","
          << bound_name(p.bound) << "\n";
    }
    return true;
}

std::vector<KernelDesc> default_bench_suite() {
    // The kernels and shapes mirror what `bench/bench_main.cpp` exercises
    // plus a handful of dossier-only kernels that aren't in the bench
    // harness yet (their FLOP/byte counts are derived from the dossier
    // math).
    std::vector<KernelDesc> v;
    auto add = [&](const std::string& name, Engine e, double flops,
                   double bytes, double dt_mult, MemTier t) {
        v.push_back({name, e, flops, bytes, dt_mult, t});
    };

    // Dense GEMM. Shapes are production-realistic (not the tiny shapes
    // bench_main.cpp uses on a developer CPU), so the predictions
    // validate the dossier's "compute-bound" claim against actual
    // Longhorn-class silicon parameters.
    {
        const int M = 4096, N = 4096, K = 4096;
        add("gemm/qkv_small", Engine::TensorUnit,
            2.0 * M * N * K,
            4.0 * (M * K + K * N + M * N),
            1.0, MemTier::HBM);
    }
    {
        const int M = 4096, N = 11008, K = 4096;
        add("gemm/mlp_up", Engine::TensorUnit,
            2.0 * M * N * K,
            4.0 * (M * K + K * N + M * N),
            1.0, MemTier::HBM);
    }
    {
        const int M = 4096, N = 4096, K = 11008;
        add("gemm/mlp_down", Engine::TensorUnit,
            2.0 * M * N * K,
            4.0 * (M * K + K * N + M * N),
            1.0, MemTier::HBM);
    }

    // RMSNorm and softmax.
    {
        const int rows = 128, dim = 4096;
        add("rmsnorm[128x4096]", Engine::NormEngine,
            3.0 * rows * dim, 2.0 * rows * dim * 4.0, 1.0, MemTier::HBM);
        add("softmax[128x4096]", Engine::SoftmaxEngine,
            5.0 * rows * dim, 2.0 * rows * dim * 4.0, 1.0, MemTier::HBM);
    }

    // Attention: causal prefill.
    {
        const int Bsz = 1, H = 8, Lq = 128, Lk = 128, d = 64;
        const double pairs = 0.5 * Bsz * H * Lq * Lk;  // half for causal
        const double flops = 4.0 * pairs * d;
        const double qn = double(Bsz) * H * Lq * d;
        const double kn = double(Bsz) * H * Lk * d;
        const double bytes = 4.0 * (qn + kn + kn + qn);
        add("flash_attn/prefill", Engine::AttentionEngine, flops, bytes,
            1.0, MemTier::HBM);
    }

    // Attention: decode (Lq=1, Lk large) — the memory-bound hot path.
    {
        const int Bsz = 1, H = 8, Lq = 1, Lk = 4096, d = 64;
        const double flops = 4.0 * Bsz * H * Lq * Lk * d;
        const double qn = double(Bsz) * H * Lq * d;
        const double kn = double(Bsz) * H * Lk * d;
        const double bytes = 4.0 * (qn + kn + kn + qn);
        add("flash_attn/decode", Engine::AttentionEngine, flops, bytes,
            1.0, MemTier::HBM);
    }

    // KV append (write-side).
    {
        const int n_kv = 8, d = 64, seq_new = 16;
        const double bytes = 2.0 * n_kv * seq_new * d * 2.0;  // K + V, FP16
        add("kv_append", Engine::KvController, 0.0, bytes, 1.0, MemTier::HBM);
    }

    // Quantized GEMM: W4A16 weights + FP32 acts.
    {
        const int M = 128, N = 4096, K = 4096, G = 128;
        const double flops = 2.0 * M * N * K;
        const double bytes_acts = 4.0 * M * K;          // FP32 acts
        const double bytes_weight = 0.5 * K * N;        // INT4 packed
        const double bytes_scale = 4.0 * (K / G) * N;   // FP32 scales
        const double bytes_out = 4.0 * M * N;
        add("gemm_w4a16[128x4096x4096,G=128]", Engine::TensorUnit,
            flops, bytes_acts + bytes_weight + bytes_scale + bytes_out,
            1.0, MemTier::HBM);
    }

    // MoE expert MLP (per-expert).
    {
        const int per_expert = 64, hidden = 4096, intermediate = 14336;
        const double flops_gate = 2.0 * per_expert * intermediate * hidden;
        const double flops_up = flops_gate;
        const double flops_down = 2.0 * per_expert * hidden * intermediate;
        const double total_flops = flops_gate + flops_up + flops_down;
        const double total_bytes =
            4.0 * (per_expert * hidden + 2.0 * hidden * intermediate +
                   per_expert * intermediate + intermediate * hidden +
                   per_expert * hidden);
        add("moe_expert_mlp[per_expert=64]", Engine::TensorUnit,
            total_flops, total_bytes, 1.0, MemTier::HBM);
    }

    // Mamba selective scan.
    {
        const int Bsz = 1, L = 2048, Di = 1024, Ds = 16;
        // ~4 FLOPs per state element per token per channel.
        const double flops = 4.0 * Bsz * L * Di * Ds;
        const double bytes = 4.0 * Bsz * L * (Di + Di + Ds + Ds + Di);
        add("selective_scan", Engine::ScanEngine, flops, bytes, 1.0,
            MemTier::HBM);
    }

    // RWKV WKV.
    {
        const int Bsz = 1, L = 2048, C = 1024;
        const double flops = 10.0 * Bsz * L * C;
        const double bytes = 4.0 * Bsz * L * C * 3.0;  // k, v, y
        add("wkv", Engine::ScanEngine, flops, bytes, 1.0, MemTier::HBM);
    }

    // Sampling pipeline.
    {
        const int vocab = 32000;
        add("sample/topk_topp", Engine::ReductionEngine,
            5.0 * vocab, 4.0 * vocab, 1.0, MemTier::HBM);
    }

    // Embedding lookup.
    {
        const int n_ids = 128, d = 4096;
        add("embedding", Engine::DmaEngine, 0.0,
            4.0 * n_ids * d, 1.0, MemTier::HBM);
    }

    // Permutation: MoE dispatch.
    {
        const int T = 64, top_k = 2, hidden = 4096;
        add("moe_dispatch", Engine::PermutationEngine,
            T * top_k, 4.0 * T * top_k * hidden, 1.0, MemTier::L2);
    }
    return v;
}

}  // namespace lh_perf
