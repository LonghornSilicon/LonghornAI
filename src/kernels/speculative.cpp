#include "kernels/speculative.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "kernels/sampling.hpp"

namespace lh {

namespace {

// Sample from a distribution given as raw probabilities (already
// normalised to sum 1, or near it). Inverse-CDF over the input row.
int32_t sample_from_probs(const float* probs, int vocab, uint64_t* rng_state) {
    const double u = uniform01(rng_state);
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[i];
        if (u < cum) return i;
    }
    return vocab - 1;  // fp drift fallback
}

// Build the corrected resampling distribution: max(0, target - draft),
// renormalised. If the residual sums to zero (exact match), fall back to
// the target distribution.
void corrected_distribution(const float* target_row, const float* draft_row,
                            int vocab, std::vector<float>& out) {
    out.assign(static_cast<size_t>(vocab), 0.0f);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float diff = target_row[i] - draft_row[i];
        const float pos = (diff > 0.0f) ? diff : 0.0f;
        out[static_cast<size_t>(i)] = pos;
        sum += pos;
    }
    if (sum <= 0.0) {
        // No positive mass left: target == draft on the supports we
        // care about. Sample from target as the fallback.
        for (int i = 0; i < vocab; ++i) out[static_cast<size_t>(i)] = target_row[i];
        return;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int i = 0; i < vocab; ++i) out[static_cast<size_t>(i)] *= inv;
}

}  // namespace

SpecVerifyResult speculative_verify(const float* draft_probs,
                                    const float* target_probs,
                                    const int32_t* draft_tokens, int K,
                                    int vocab, uint64_t* rng_state) {
    SpecVerifyResult r;
    std::vector<float> resampled(static_cast<size_t>(vocab));

    for (int k = 0; k < K; ++k) {
        const int32_t x = draft_tokens[k];
        const float* drow = draft_probs + static_cast<int64_t>(k) * vocab;
        const float* trow = target_probs + static_cast<int64_t>(k) * vocab;
        const float dp = drow[x];
        const float tp = trow[x];
        // Acceptance probability: min(1, tp / dp). When dp == 0, the
        // draft assigned zero mass to its own choice — pathological;
        // accept iff tp > 0.
        float a;
        if (dp <= 0.0f) {
            a = (tp > 0.0f) ? 1.0f : 0.0f;
        } else {
            a = (tp >= dp) ? 1.0f : (tp / dp);
        }
        const float u = static_cast<float>(uniform01(rng_state));
        if (u < a) {
            ++r.n_accepted;
            continue;
        }
        // Reject: sample from corrected distribution.
        corrected_distribution(trow, drow, vocab, resampled);
        r.bonus_token = sample_from_probs(resampled.data(), vocab, rng_state);
        return r;
    }
    // All K accepted: sample from the (K+1)th target row.
    r.bonus_token = sample_from_probs(
        target_probs + static_cast<int64_t>(K) * vocab, vocab, rng_state);
    return r;
}

void build_tree_attention_bias(const int32_t* parents, int n_nodes,
                               int n_history, float* bias_out) {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    const int seq_k_total = n_history + n_nodes;

    // Precompute ancestor sets via parent walk. Node q attends to
    // history (always) and to itself + any ancestor.
    for (int q = 0; q < n_nodes; ++q) {
        float* row = bias_out + static_cast<int64_t>(q) * seq_k_total;
        // History positions: bias 0 (visible).
        for (int h = 0; h < n_history; ++h) row[h] = 0.0f;
        // Tree positions: -inf by default.
        for (int j = 0; j < n_nodes; ++j) row[n_history + j] = kNegInf;

        // Walk q -> root, marking visible.
        int p = q;
        while (p >= 0) {
            row[n_history + p] = 0.0f;
            const int next = parents[p];
            if (next == p) break;  // self-loop sentinel
            p = next;
        }
    }
}

}  // namespace lh
