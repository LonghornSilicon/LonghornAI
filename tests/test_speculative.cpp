#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/sampling.hpp"
#include "kernels/speculative.hpp"
#include "test_util.hpp"

namespace {

// Make a simple test distribution from soft logits.
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> p(logits.size());
    float m = logits[0];
    for (float l : logits) m = std::max(m, l);
    float s = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] - m);
        s += p[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) p[i] /= s;
    return p;
}

}  // namespace

TEST(Speculative, AllAcceptedWhenDraftEqualsTarget) {
    constexpr int K = 4, vocab = 8;
    auto base = softmax(lh_test::random_vector(vocab, 8000));
    std::vector<float> draft(K * vocab);
    std::vector<float> target((K + 1) * vocab);
    for (int k = 0; k < K; ++k) {
        std::copy(base.begin(), base.end(),
                  draft.begin() + static_cast<int64_t>(k) * vocab);
        std::copy(base.begin(), base.end(),
                  target.begin() + static_cast<int64_t>(k) * vocab);
    }
    std::copy(base.begin(), base.end(),
              target.begin() + static_cast<int64_t>(K) * vocab);

    // Pick draft tokens with non-zero probability (any).
    std::vector<int32_t> draft_tokens(K);
    for (int k = 0; k < K; ++k) draft_tokens[k] = 0;

    uint64_t rng = 0x12345;
    auto r = lh::speculative_verify(draft.data(), target.data(),
                                    draft_tokens.data(), K, vocab, &rng);
    EXPECT_EQ(r.n_accepted, K);
    EXPECT_GE(r.bonus_token, 0);
    EXPECT_LT(r.bonus_token, vocab);
}

TEST(Speculative, RejectionPreservesTargetMarginal) {
    // Statistical test: when draft and target differ, the *marginal*
    // distribution of the first emitted token (across many trials) must
    // match the target's first-position distribution exactly. This is
    // the correctness guarantee of speculative decoding.
    constexpr int K = 1, vocab = 4;
    // Target favours token 1; draft favours token 0.
    std::vector<float> target = {0.1f, 0.6f, 0.2f, 0.1f, /*K-th row*/ 0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<float> draft  = {0.6f, 0.1f, 0.2f, 0.1f};

    constexpr int N = 30000;
    std::vector<int> hits(vocab, 0);
    uint64_t rng = 0xfeedface;
    for (int trial = 0; trial < N; ++trial) {
        // Sample a draft token from the draft distribution (this is what
        // the draft model would do at inference time).
        const int32_t dt = lh::softmax_sample(
            // log probs — softmax_sample takes logits, but log of probs
            // works because softmax(log p) = p.
            std::vector<float>{std::log(draft[0]), std::log(draft[1]),
                                std::log(draft[2]), std::log(draft[3])}.data(),
            vocab, &rng);
        const int32_t draft_tokens_arr[1] = {dt};
        auto r = lh::speculative_verify(draft.data(), target.data(),
                                        draft_tokens_arr, K, vocab, &rng);
        // The first emitted token is dt if accepted, else r.bonus_token.
        const int32_t first =
            (r.n_accepted >= 1) ? dt : r.bonus_token;
        ++hits[static_cast<size_t>(first)];
    }
    for (int i = 0; i < vocab; ++i) {
        const double emp = static_cast<double>(hits[i]) / N;
        EXPECT_NEAR(emp, static_cast<double>(target[i]), 0.025)
            << "i=" << i << " emp=" << emp << " target=" << target[i];
    }
}

TEST(Speculative, TreeMaskChainEqualsCausal) {
    // A chain tree (parent[i] = i-1) should produce a causal mask:
    //   node q attends to history + nodes {0..q}.
    constexpr int n_nodes = 4, n_history = 2;
    std::vector<int32_t> parents = {-1, 0, 1, 2};
    const int seq_k = n_history + n_nodes;
    std::vector<float> bias(n_nodes * seq_k);
    lh::build_tree_attention_bias(parents.data(), n_nodes, n_history,
                                  bias.data());

    const float ninf = -std::numeric_limits<float>::infinity();
    for (int q = 0; q < n_nodes; ++q) {
        // History always visible.
        for (int h = 0; h < n_history; ++h) {
            EXPECT_EQ(bias[q * seq_k + h], 0.0f);
        }
        // Causal over the tree positions.
        for (int j = 0; j < n_nodes; ++j) {
            const float v = bias[q * seq_k + n_history + j];
            if (j <= q) EXPECT_EQ(v, 0.0f) << "q=" << q << " j=" << j;
            else EXPECT_EQ(v, ninf) << "q=" << q << " j=" << j;
        }
    }
}

TEST(Speculative, TreeMaskForkBlocksCrossBranch) {
    // Tree:
    //         0 (root)
    //        / \
    //       1   2
    //      / \   \
    //     3   4   5
    // Node 3 must NOT see {2, 4, 5}; node 5 must NOT see {1, 3, 4}.
    constexpr int n_nodes = 6, n_history = 0;
    std::vector<int32_t> parents = {-1, 0, 0, 1, 1, 2};
    std::vector<float> bias(n_nodes * n_nodes);
    lh::build_tree_attention_bias(parents.data(), n_nodes, n_history,
                                  bias.data());
    const float ninf = -std::numeric_limits<float>::infinity();

    // Node 3 sees {0, 1, 3}.
    EXPECT_EQ(bias[3 * n_nodes + 0], 0.0f);
    EXPECT_EQ(bias[3 * n_nodes + 1], 0.0f);
    EXPECT_EQ(bias[3 * n_nodes + 2], ninf);
    EXPECT_EQ(bias[3 * n_nodes + 3], 0.0f);
    EXPECT_EQ(bias[3 * n_nodes + 4], ninf);
    EXPECT_EQ(bias[3 * n_nodes + 5], ninf);

    // Node 5 sees {0, 2, 5}.
    EXPECT_EQ(bias[5 * n_nodes + 0], 0.0f);
    EXPECT_EQ(bias[5 * n_nodes + 1], ninf);
    EXPECT_EQ(bias[5 * n_nodes + 2], 0.0f);
    EXPECT_EQ(bias[5 * n_nodes + 3], ninf);
    EXPECT_EQ(bias[5 * n_nodes + 4], ninf);
    EXPECT_EQ(bias[5 * n_nodes + 5], 0.0f);
}
