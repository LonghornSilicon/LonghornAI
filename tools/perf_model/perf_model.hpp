// LonghornAI — analytical Longhorn performance model.
//
// Phase 9 Stage A simulator. Takes a kernel description plus a
// `LonghornConfig` (per-engine peak FLOPS, memory tier bandwidths) and
// returns a predicted execution time plus roofline classification.
//
// This is *not* cycle-accurate. It models the dominant roof per engine
// (compute or memory bandwidth) and reports the bottleneck. Sufficient
// for architectural validation of the kernel inventory; Stage B (FPGA)
// adds pipeline-stall and NoC-routing detail.
//
// The model deliberately mirrors the Phase 8 dossier framing: every
// prediction here can be cross-checked against
// `docs/silicon/kernels/<name>.md` and `kernel_engine_map.md`.
#ifndef LONGHORNAI_TOOLS_PERF_MODEL_HPP
#define LONGHORNAI_TOOLS_PERF_MODEL_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace lh_perf {

// Engine identifiers match the blocks in `architecture.md` §3-7.
enum class Engine {
    TensorUnit,            // dense / batched / grouped GEMM, MoE expert MLP
    AttentionEngine,       // SDPA / Flash / FlashDecoding / paged
    KvController,          // KV append, dequant on read
    VectorUnit,            // pointwise, activations, RoPE, conv1d, sampling filters
    NormEngine,            // RMSNorm / LayerNorm
    SoftmaxEngine,         // standalone softmax (attention's softmax fuses inside Attn)
    ReductionEngine,       // sum / max / mean / argmax / topk
    PermutationEngine,     // MoE dispatch/combine, transpose, gather/scatter
    ScanEngine,            // Mamba SSM, RWKV WKV, linear attention
    DmaEngine,             // pure data motion (embedding, transform)
    HostScheduler,         // cold-path setup (NTK/YaRN, tree mask)
};

const char* engine_name(Engine e);

// Bound class — same definition as bench_util.hpp's RooflineClass but
// duplicated here to keep the perf model standalone.
enum class Bound { Memory, Balanced, Compute };
const char* bound_name(Bound b);

// Memory tier the kernel's traffic lives in.
enum class MemTier {
    HBM,         // off-chip primary memory (server)
    LPDDR,       // off-chip primary memory (edge)
    L2,          // on-chip shared SRAM
    L1,          // per-tile SRAM
};

// Per-kernel description. `name` is human-readable; `engine` and
// `mem_tier` drive which roof applies. `flops` and `bytes` are per
// invocation (one tile, one row, etc. — caller sets the granularity).
// `dtype_throughput_mult` lets the caller signal that the kernel runs
// at 2× FP16 (INT8) or 4× FP16 (INT4) on the Tensor Unit.
struct KernelDesc {
    std::string name;
    Engine engine = Engine::TensorUnit;
    double flops = 0.0;
    double bytes = 0.0;
    double dtype_throughput_mult = 1.0;
    MemTier mem_tier = MemTier::HBM;
};

// Hardware configuration (`LonghornConfig`). Picks edge vs server.
struct LonghornConfig {
    // Tensor unit peak per tile, in FP16 ops/sec. INT8 doubles, INT4
    // quadruples, FP32 halves (informational; the dtype mult is on the
    // KernelDesc).
    double tensor_peak_fp16 = 4.0e12;       // 4 TFLOPS (edge default)
    int    tensor_tiles = 1;                 // 8 in server profile
    double vector_peak = 256.0e9;            // 256 GFLOPS per tile
    double norm_peak = 32.0e9;               // 32 G reductions/s per tile
    double softmax_peak = 32.0e9;
    double reduction_peak = 32.0e9;
    double scan_peak = 128.0e9;              // per tile
    double permute_peak = 256.0e9;           // per chip (Permutation Engine sized for L2 BW)

    // Memory tier effective bandwidth (peak * utilisation).
    double bw_l1_per_tile = 1.0e12;          // 1 TB/s
    double bw_l2 = 500.0e9;
    double bw_hbm = 100.0e9;                 // edge default = LPDDR5X
    double bw_lpddr = 100.0e9;

    // Whether the host runs cold-path setup (NTK / YaRN / tree mask
    // build). When true, host_peak determines runtime; when false the
    // kernel is treated as zero-cost (already cached).
    double host_peak = 1.0e10;

    // Convenience constructors.
    static LonghornConfig edge();
    static LonghornConfig server();
};

// Prediction result.
struct Prediction {
    std::string name;
    Engine engine;
    double t_compute_s = 0.0;
    double t_memory_s = 0.0;
    double t_predicted_s = 0.0;     // max of the two
    double ai_flops_per_byte = 0.0;
    Bound bound = Bound::Balanced;
};

// Predict execution time for one kernel.
Prediction predict(const KernelDesc& k, const LonghornConfig& cfg);

// Predict a list and return per-kernel predictions.
std::vector<Prediction> predict_all(const std::vector<KernelDesc>& kernels,
                                    const LonghornConfig& cfg);

// CSV writer matching the bench harness's --roofline format so the
// outputs are diff-able.
bool write_predictions_csv(const std::string& path,
                           const std::vector<Prediction>& preds);

// Bench-shape suite that cross-references the Phase 8 dossiers.
// Ships with the simulator so callers don't have to hand-roll it.
std::vector<KernelDesc> default_bench_suite();

}  // namespace lh_perf

#endif  // LONGHORNAI_TOOLS_PERF_MODEL_HPP
