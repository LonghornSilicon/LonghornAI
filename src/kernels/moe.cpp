#include "kernels/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "kernels/gemm.hpp"
#include "kernels/reduction.hpp"
#include "kernels/softmax.hpp"

namespace lh {

namespace {

// One stable per-row softmax + top-k pass. Avoids materialising a full
// [n_tokens, n_experts] softmax buffer — we only need the top-k values
// and their indices to drive routing, and softmax + top-k commute under
// monotone transforms (top-k by score == top-k by softmax(score)).
void softmax_topk_row(const float* logits, int32_t* ids_out, float* gates_out,
                      int n_experts, int top_k, bool renorm) {
    using Pair = std::pair<float, int32_t>;
    auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    std::vector<Pair> heap;
    heap.reserve(static_cast<size_t>(top_k));

    // Top-k on raw logits (== top-k on softmax probabilities).
    for (int e = 0; e < n_experts; ++e) {
        const Pair p{logits[e], e};
        if (static_cast<int>(heap.size()) < top_k) {
            heap.push_back(p);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (logits[e] > heap.front().first ||
                   (logits[e] == heap.front().first &&
                    e < heap.front().second)) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = p;
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    std::sort(heap.begin(), heap.end(),
              [](const Pair& a, const Pair& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });

    // Stable softmax over the survivors only when `renorm` is true; then
    // the gates form a probability distribution over the chosen experts.
    // Otherwise compute softmax over all logits and read the top-k values.
    if (renorm) {
        float m = -std::numeric_limits<float>::infinity();
        for (const auto& p : heap) if (p.first > m) m = p.first;
        float sum = 0.0f;
        std::vector<float> exps(static_cast<size_t>(top_k));
        for (int i = 0; i < top_k; ++i) {
            exps[static_cast<size_t>(i)] = std::exp(heap[i].first - m);
            sum += exps[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < top_k; ++i) {
            ids_out[i] = heap[i].second;
            gates_out[i] = exps[i] * inv;
        }
    } else {
        float m = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < n_experts; ++e) if (logits[e] > m) m = logits[e];
        float sum = 0.0f;
        for (int e = 0; e < n_experts; ++e) sum += std::exp(logits[e] - m);
        const float inv = 1.0f / sum;
        for (int i = 0; i < top_k; ++i) {
            ids_out[i] = heap[i].second;
            gates_out[i] = std::exp(heap[i].first - m) * inv;
        }
    }
}

}  // namespace

void moe_router(const float* hidden, const float* router_weight,
                int32_t* expert_ids, float* gate_weights,
                const MoEConfig& cfg) {
    const int T = cfg.n_tokens;
    const int H = cfg.hidden_dim;
    const int E = cfg.n_experts;
    const int K = cfg.top_k;

    // logits = hidden [T, H] @ router_weight [H, E]  ->  [T, E].
    std::vector<float> logits(static_cast<size_t>(T) * E, 0.0f);
    gemm(hidden, router_weight, logits.data(), T, E, H);

    for (int t = 0; t < T; ++t) {
        softmax_topk_row(logits.data() + static_cast<int64_t>(t) * E,
                         expert_ids + static_cast<int64_t>(t) * K,
                         gate_weights + static_cast<int64_t>(t) * K,
                         E, K, cfg.renorm_gates);
    }
}

void moe_dispatch(const float* hidden, const int32_t* expert_ids,
                  int32_t* expert_offsets, int32_t* token_idx,
                  int32_t* slot_idx, float* expert_inputs,
                  const MoEConfig& cfg) {
    const int T = cfg.n_tokens;
    const int H = cfg.hidden_dim;
    const int E = cfg.n_experts;
    const int K = cfg.top_k;

    // 1. Count tokens per expert.
    std::vector<int32_t> counts(static_cast<size_t>(E), 0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const int32_t e = expert_ids[t * K + k];
            ++counts[static_cast<size_t>(e)];
        }
    }
    // 2. Exclusive prefix sum -> expert_offsets[e+1].
    expert_offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        expert_offsets[e + 1] = expert_offsets[e] + counts[static_cast<size_t>(e)];
    }
    // 3. Place each (token, slot) row into its expert lane. We reuse
    //    `counts` as a write cursor so we can fill the dispatched buffer
    //    in a single pass over the activation data.
    std::fill(counts.begin(), counts.end(), 0);
    for (int t = 0; t < T; ++t) {
        const float* src = hidden + static_cast<int64_t>(t) * H;
        for (int k = 0; k < K; ++k) {
            const int32_t e = expert_ids[t * K + k];
            const int64_t pos =
                expert_offsets[e] + counts[static_cast<size_t>(e)]++;
            token_idx[pos] = t;
            slot_idx[pos] = k;
            float* dst = expert_inputs + pos * H;
            std::memcpy(dst, src, static_cast<size_t>(H) * sizeof(float));
        }
    }
}

namespace {

inline float silu_scalar(float v) {
    return v / (1.0f + std::exp(-v));
}

}  // namespace

void moe_expert_mlp(const float* expert_inputs, const int32_t* expert_offsets,
                    const float* W_gate, const float* W_up,
                    const float* W_down, float* expert_outputs,
                    const MoEConfig& cfg) {
    const int H = cfg.hidden_dim;
    const int I = cfg.intermediate_dim;
    const int E = cfg.n_experts;

    // Per-expert: run gate + up GEMMs, apply SwiGLU, run down GEMM.
    // We use scratch buffers for gate_proj, up_proj, hidden_act so the
    // downstream gemm has a contiguous input.
    int max_count = 0;
    for (int e = 0; e < E; ++e) {
        const int n = expert_offsets[e + 1] - expert_offsets[e];
        if (n > max_count) max_count = n;
    }
    if (max_count == 0) return;
    std::vector<float> gate_proj(static_cast<size_t>(max_count) * I);
    std::vector<float> up_proj(static_cast<size_t>(max_count) * I);
    std::vector<float> act(static_cast<size_t>(max_count) * I);

    for (int e = 0; e < E; ++e) {
        const int start = expert_offsets[e];
        const int end = expert_offsets[e + 1];
        const int n = end - start;
        if (n == 0) continue;

        const float* xe = expert_inputs + static_cast<int64_t>(start) * H;
        const float* Wg = W_gate +
                          static_cast<int64_t>(e) * H * I;
        const float* Wu = W_up +
                          static_cast<int64_t>(e) * H * I;
        const float* Wd = W_down +
                          static_cast<int64_t>(e) * I * H;

        gemm(xe, Wg, gate_proj.data(), n, I, H);
        gemm(xe, Wu, up_proj.data(), n, I, H);
        for (int t = 0; t < n; ++t) {
            const float* g = gate_proj.data() + static_cast<int64_t>(t) * I;
            const float* u = up_proj.data() + static_cast<int64_t>(t) * I;
            float* a = act.data() + static_cast<int64_t>(t) * I;
            for (int i = 0; i < I; ++i) a[i] = silu_scalar(g[i]) * u[i];
        }
        gemm(act.data(), Wd,
             expert_outputs + static_cast<int64_t>(start) * H, n, H, I);
    }
}

void moe_combine(const float* expert_outputs, const int32_t* /*expert_offsets*/,
                 const int32_t* token_idx, const int32_t* slot_idx,
                 const float* gate_weights, float* output,
                 const MoEConfig& cfg) {
    const int T = cfg.n_tokens;
    const int H = cfg.hidden_dim;
    const int K = cfg.top_k;
    const int total_slots = T * K;

    std::memset(output, 0,
                static_cast<size_t>(T) * H * sizeof(float));

    // One pass over the dispatched output buffer: for each slot, scatter
    // gate * expert_output into the corresponding token's row. This is
    // the "single shuffle pass" combine acceptance.
    for (int s = 0; s < total_slots; ++s) {
        const int32_t t = token_idx[s];
        const int32_t k = slot_idx[s];
        const float gate = gate_weights[t * K + k];
        const float* y = expert_outputs + static_cast<int64_t>(s) * H;
        float* o = output + static_cast<int64_t>(t) * H;
        for (int i = 0; i < H; ++i) o[i] += gate * y[i];
    }
}

void moe_forward(const float* hidden, const float* router_weight,
                 const float* W_gate, const float* W_up, const float* W_down,
                 float* output, const MoEConfig& cfg) {
    const int T = cfg.n_tokens;
    const int K = cfg.top_k;
    const int E = cfg.n_experts;

    std::vector<int32_t> expert_ids(static_cast<size_t>(T) * K);
    std::vector<float> gates(static_cast<size_t>(T) * K);
    moe_router(hidden, router_weight, expert_ids.data(), gates.data(), cfg);

    std::vector<int32_t> offsets(static_cast<size_t>(E + 1), 0);
    std::vector<int32_t> tok_idx(static_cast<size_t>(T) * K);
    std::vector<int32_t> sl_idx(static_cast<size_t>(T) * K);
    std::vector<float> exp_in(static_cast<size_t>(T) * K * cfg.hidden_dim);
    moe_dispatch(hidden, expert_ids.data(), offsets.data(), tok_idx.data(),
                 sl_idx.data(), exp_in.data(), cfg);

    std::vector<float> exp_out(static_cast<size_t>(T) * K * cfg.hidden_dim);
    moe_expert_mlp(exp_in.data(), offsets.data(), W_gate, W_up, W_down,
                   exp_out.data(), cfg);

    moe_combine(exp_out.data(), offsets.data(), tok_idx.data(), sl_idx.data(),
                gates.data(), output, cfg);
}

}  // namespace lh
