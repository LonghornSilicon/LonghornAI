// LonghornAI — Mixture-of-Experts kernels.
//
// Sparse-MLP routing kernels: router, dispatch, expert FFN, combine. The
// reference layout follows Mixtral / DeepSeek:
//
//   Per-token data flow:
//     hidden -> router(Linear + softmax) -> top-k (expert_id, gate)
//     for k in top_k:                        per-token expert MLP
//        x_e = expert_input(token, k)        SwiGLU(W_gate, W_up, W_down)
//        y_e = expert_mlp(e, x_e)
//     out = sum_k gate[k] * y_e
//
// Storage:
//   hidden:        [n_tokens, hidden_dim]
//   router_weight: [hidden_dim, n_experts]      (Linear weight)
//   W_gate:        [n_experts, hidden_dim, intermediate]
//   W_up:          [n_experts, hidden_dim, intermediate]
//   W_down:        [n_experts, intermediate, hidden_dim]
//
// The dispatched layout shuffles tokens into expert-major order so each
// expert's MLP becomes a single contiguous GEMM (and the whole layer is a
// grouped GEMM). Dispatch + combine are each a single pass over the
// activation buffer (counting sort + indirected scatter / weighted gather),
// matching the "single shuffle pass" acceptance criterion in PLAN.md §5.
//
// Out of scope here:
//   - Quantized expert weights (compose by replacing the expert-MLP path
//     with `gemm_w4a16_groupwise` once weights are stored that way).
//   - Shared-expert designs (DeepSeek): same shuffle layout, plus an
//     unconditional dense MLP added to the combine output.
//   - Cross-device expert parallelism (Phase 9).
#ifndef LONGHORNAI_KERNELS_MOE_HPP
#define LONGHORNAI_KERNELS_MOE_HPP

#include <cstdint>

namespace lh {

struct MoEConfig {
    int n_tokens = 0;
    int hidden_dim = 0;
    int intermediate_dim = 0;
    int n_experts = 0;
    int top_k = 1;
    // If true, renormalise the chosen top-k gate values to sum to 1 per
    // token (Mixtral default; DeepSeek does the same). When false the raw
    // softmax probabilities flow through.
    bool renorm_gates = true;
};

// Router: hidden @ router_weight -> softmax -> top-k.
//   hidden:        fp32 [n_tokens, hidden_dim]
//   router_weight: fp32 [hidden_dim, n_experts]
//   expert_ids:    int32 [n_tokens, top_k]   (descending by score)
//   gate_weights:  fp32  [n_tokens, top_k]   (post-softmax, post-renorm)
//
// Tie-breaking on equal scores follows `topk` in `reduction.hpp`: lowest
// index wins.
void moe_router(const float* hidden, const float* router_weight,
                int32_t* expert_ids, float* gate_weights,
                const MoEConfig& cfg);

// Dispatch: build the per-expert token list and scatter activations into
// expert-major order. Scratch buffers expected:
//
//   expert_offsets[n_experts + 1]  (output) prefix sum of per-expert counts
//   token_idx[n_tokens * top_k]    (output) source token index per slot
//   slot_idx[n_tokens * top_k]     (output) original top-k position
//   expert_inputs[n_tokens * top_k, hidden] (output) shuffled inputs
//
// The dispatched buffer is laid out as: for each expert e, a contiguous
// block of `expert_offsets[e+1] - expert_offsets[e]` rows, each a copy of
// the routed token's hidden state. Slot order within an expert preserves
// token order to keep the layout deterministic.
void moe_dispatch(const float* hidden, const int32_t* expert_ids,
                  int32_t* expert_offsets, int32_t* token_idx,
                  int32_t* slot_idx, float* expert_inputs,
                  const MoEConfig& cfg);

// Per-expert SwiGLU MLP via grouped GEMM:
//   y[e][t] = (silu(x[e][t] @ W_gate[e]) * (x[e][t] @ W_up[e])) @ W_down[e]
//
//   expert_inputs:  fp32 [total_slots, hidden_dim]   (from moe_dispatch)
//   expert_offsets: int32 [n_experts + 1]
//   W_gate:         fp32 [n_experts, hidden_dim, intermediate_dim]
//   W_up:           fp32 [n_experts, hidden_dim, intermediate_dim]
//   W_down:         fp32 [n_experts, intermediate_dim, hidden_dim]
//   expert_outputs: fp32 [total_slots, hidden_dim]   (output)
//
// Empty experts (offsets[e] == offsets[e+1]) are skipped.
void moe_expert_mlp(const float* expert_inputs, const int32_t* expert_offsets,
                    const float* W_gate, const float* W_up,
                    const float* W_down, float* expert_outputs,
                    const MoEConfig& cfg);

// Combine: gate-weighted gather of expert outputs back into per-token
// outputs.  output[t] = sum_k gate[t, k] * expert_outputs[slot(t, k)].
// `slot(t, k)` is recovered from (token_idx, slot_idx, expert_offsets).
void moe_combine(const float* expert_outputs, const int32_t* expert_offsets,
                 const int32_t* token_idx, const int32_t* slot_idx,
                 const float* gate_weights, float* output,
                 const MoEConfig& cfg);

// Convenience: full Mixtral/DeepSeek-style MoE block (router + dispatch +
// expert MLP + combine). Allocates scratch internally; for production
// hot paths call the four primitives above with reusable scratch buffers.
void moe_forward(const float* hidden, const float* router_weight,
                 const float* W_gate, const float* W_up, const float* W_down,
                 float* output, const MoEConfig& cfg);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_MOE_HPP
