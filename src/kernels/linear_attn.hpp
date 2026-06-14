// LonghornAI — causal linear attention.
//
// Replaces softmax(QKᵀ)V with the kernel-feature form:
//   y_t = (φ(Q_t)ᵀ @ S_t) / Z_t
// where:
//   S_t = sum_{i <= t} φ(K_i) ⊗ V_i   ∈ R^{d_feat × d_v}
//   Z_t = φ(Q_t)ᵀ @ sum_{i <= t} φ(K_i)   (denominator; can be 1 if
//        the feature map already sums to 1, e.g. ELU+1 or random
//        Performer features chosen so).
//
// Q and K here are *already* feature-mapped (φ applied by the caller).
// This keeps the kernel agnostic to which linear-attention recipe is in
// use — Performer (random features), ELU+1, identity, exp-shifted, etc.
// The caller chooses φ; we provide the recurrence.
//
// Shape:
//   Q, K: fp32 [batch, n_heads, seq, d_feat]
//   V:    fp32 [batch, n_heads, seq, d_v]
//   y:    fp32 [batch, n_heads, seq, d_v]
//
// State per (batch, head): S = [d_feat, d_v], z = [d_feat]. Both grow by
// one rank-1 update per token. Foundation for RetNet's parallel /
// recurrent / chunkwise duality and for general kernel-attention models.
#ifndef LONGHORNAI_KERNELS_LINEAR_ATTN_HPP
#define LONGHORNAI_KERNELS_LINEAR_ATTN_HPP

#include <cstdint>

namespace lh {

struct LinearAttnConfig {
    int batch = 1;
    int n_heads = 1;
    int seq = 1;
    int d_feat = 1;
    int d_v = 1;
    bool normalize = true;  // divide by Z; if false, the bare numerator is
                            // returned (RetNet-style retention does this)
};

void linear_attention(const float* Q, const float* K, const float* V,
                      float* y, const LinearAttnConfig& cfg);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_LINEAR_ATTN_HPP
