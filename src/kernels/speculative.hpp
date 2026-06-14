// LonghornAI — speculative decoding kernels.
//
// Two pieces:
//
// 1. `speculative_verify`: the canonical Leviathan/Chen 2023 accept/reject
//    rule. Given K draft tokens with their draft-model probabilities and
//    the target-model probabilities at the same K positions (plus
//    position K+1 for the bonus), return the longest accepted prefix and
//    a corrected next token sampled from the right distribution. The
//    output distribution provably matches sampling K+1 tokens directly
//    from the target model.
//
// 2. `build_tree_attention_bias`: tree-of-candidates attention mask
//    builder. Medusa, Eagle, and tree-decoding all run M candidate
//    continuations through the target model in a single forward pass by
//    arranging the candidates as nodes in a tree and feeding them as a
//    single Q sequence with a per-(query, key) mask that forbids
//    attention from a node to anything not on its root-to-self path.
//    The output is a fp32 additive bias that drops into
//    `AttnConfig::bias`.
#ifndef LONGHORNAI_KERNELS_SPECULATIVE_HPP
#define LONGHORNAI_KERNELS_SPECULATIVE_HPP

#include <cstdint>

namespace lh {

struct SpecVerifyResult {
    int n_accepted = 0;        // number of draft tokens accepted, in [0, K]
    int32_t bonus_token = 0;   // sampled corrected/free token
};

// Verify K draft tokens against the target distribution.
//
//   draft_probs:   fp32 [K, vocab]   draft model's probability rows
//   target_probs:  fp32 [K + 1, vocab]
//                                    target model's probability rows; the
//                                    extra row is read iff all K accepted
//   draft_tokens:  int32 [K]
//   rng_state:     splitmix64 state (mutated)
//
// Algorithm (positions are 0-indexed into the K draft tokens):
//   for k in 0..K-1:
//     x = draft_tokens[k]
//     a = min(1, target[k][x] / draft[k][x])
//     if uniform() < a: accept and continue
//     else:
//       sample bonus from `max(0, target[k] - draft[k])` (renormalised)
//       return (k, bonus)
//   all accepted -> sample bonus from target[K]; return (K, bonus)
SpecVerifyResult speculative_verify(const float* draft_probs,
                                    const float* target_probs,
                                    const int32_t* draft_tokens,
                                    int K, int vocab,
                                    uint64_t* rng_state);

// Build the [n_nodes, n_nodes + n_history] additive attention bias for a
// tree of candidate continuations.
//
//   parents:      int32 [n_nodes]  parent index per node; root has -1.
//                                  Nodes must appear in topological order
//                                  (parent index < child index).
//   n_history:    int               number of preceding cache positions
//                                  every node attends to (the request's
//                                  KV-cache prefix length).
//   bias_out:     fp32 [n_nodes, n_history + n_nodes]
//
// Layout of the bias's `seq_k` axis:
//   [0, n_history)            : history; every node attends here (bias 0)
//   [n_history, n_history+i]  : tree node i; node q attends iff i is on
//                                q's root-to-self path (q's ancestors or
//                                q itself); else -inf.
void build_tree_attention_bias(const int32_t* parents, int n_nodes,
                               int n_history, float* bias_out);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_SPECULATIVE_HPP
