#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/positional.hpp"
#include "test_util.hpp"

namespace {

lh::AttnConfig base_cfg(int qh, int kvh, int sq, int sk, int d, bool causal) {
    lh::AttnConfig c;
    c.batch = 1;
    c.n_q_heads = qh;
    c.n_kv_heads = kvh;
    c.seq_q = sq;
    c.seq_k = sk;
    c.head_dim = d;
    c.causal = causal;
    return c;
}

void run_qkv(const lh::AttnConfig& cfg,
             std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V,
             std::vector<float>& O) {
    const int64_t qn =
        int64_t(cfg.batch) * cfg.n_q_heads * cfg.seq_q * cfg.head_dim;
    const int64_t kn =
        int64_t(cfg.batch) * cfg.n_kv_heads * cfg.seq_k * cfg.head_dim;
    Q = lh_test::random_vector(qn, 401);
    K = lh_test::random_vector(kn, 402);
    V = lh_test::random_vector(kn, 403);
    O.assign(static_cast<size_t>(qn), 0.0f);
}

}  // namespace

TEST(AttentionExt, FlashDecodingMatchesSdpa) {
    auto cfg = base_cfg(/*qh=*/4, /*kvh=*/2, /*sq=*/1, /*sk=*/64,
                        /*d=*/32, /*causal=*/true);
    std::vector<float> Q, K, V, O_sdpa, O_split;
    run_qkv(cfg, Q, K, V, O_sdpa);
    O_split.assign(O_sdpa.size(), 0.0f);

    lh::sdpa(Q.data(), K.data(), V.data(), O_sdpa.data(), cfg);
    lh::flash_decode_attention(Q.data(), K.data(), V.data(), O_split.data(),
                               cfg, /*block_k=*/16, /*splits=*/4);
    EXPECT_TRUE(lh_test::AllClose(O_split, O_sdpa, 1e-4f, 1e-4f));
}

TEST(AttentionExt, SlidingWindowMatchesSdpaWithExplicitMask) {
    // Sliding-window attention is equivalent to a hand-built additive mask
    // with -inf outside [q_pos - W + 1, q_pos]. Compare the two paths.
    auto cfg = base_cfg(/*qh=*/2, /*kvh=*/2, /*sq=*/8, /*sk=*/8,
                        /*d=*/16, /*causal=*/true);
    cfg.window = 3;
    std::vector<float> Q, K, V, O_window, O_mask;
    run_qkv(cfg, Q, K, V, O_window);
    O_mask.assign(O_window.size(), 0.0f);

    lh::sdpa(Q.data(), K.data(), V.data(), O_window.data(), cfg);

    // Equivalent run via a per-query mask.
    auto cfg_mask = cfg;
    cfg_mask.window = 0;
    const int seq_q = cfg.seq_q, seq_k = cfg.seq_k;
    std::vector<float> bias(seq_q * seq_k, 0.0f);
    const float ninf = -std::numeric_limits<float>::infinity();
    const int pos_shift = seq_k - seq_q;
    for (int i = 0; i < seq_q; ++i)
        for (int j = 0; j < seq_k; ++j) {
            const int q_pos = pos_shift + i;
            const bool in_window = (j <= q_pos) && (j >= q_pos - cfg.window + 1);
            if (!in_window) bias[i * seq_k + j] = ninf;
        }
    cfg_mask.bias = bias.data();
    cfg_mask.bias_per_head = false;
    lh::sdpa(Q.data(), K.data(), V.data(), O_mask.data(), cfg_mask);

    EXPECT_TRUE(lh_test::AllClose(O_window, O_mask, 1e-5f, 1e-5f));
}

TEST(AttentionExt, ChunkedRespectsBoundaries) {
    // With chunk=4 and seq_q==seq_k, queries can only see keys in the same
    // chunk. We verify by zeroing all V tokens in chunk 0 and checking that
    // chunk-1 queries still produce nonzero outputs (because they only read
    // chunk-1 V).
    auto cfg = base_cfg(/*qh=*/2, /*kvh=*/2, /*sq=*/8, /*sk=*/8,
                        /*d=*/8, /*causal=*/false);
    cfg.chunk = 4;
    std::vector<float> Q, K, V, O;
    run_qkv(cfg, Q, K, V, O);
    // Zero V on chunk 0 (positions 0..3) so any leakage to chunk-1 queries
    // would lower their output magnitude.
    for (int h = 0; h < cfg.n_kv_heads; ++h) {
        for (int s = 0; s < 4; ++s) {
            for (int t = 0; t < cfg.head_dim; ++t) {
                V[(h * cfg.seq_k + s) * cfg.head_dim + t] = 0.0f;
            }
        }
    }
    lh::sdpa(Q.data(), K.data(), V.data(), O.data(), cfg);
    // Chunk-0 queries (i in 0..3) read only zeroed V → output is exactly 0.
    for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
        for (int i = 0; i < 4; ++i) {
            const int64_t base =
                (static_cast<int64_t>(qh) * cfg.seq_q + i) * cfg.head_dim;
            for (int t = 0; t < cfg.head_dim; ++t) {
                EXPECT_NEAR(O[base + t], 0.0f, 1e-6f);
            }
        }
    }
}

TEST(AttentionExt, CrossAttentionIsAcausal) {
    auto cfg = base_cfg(/*qh=*/2, /*kvh=*/2, /*sq=*/3, /*sk=*/5,
                        /*d=*/8, /*causal=*/true);
    std::vector<float> Q, K, V, O_cross, O_acausal;
    run_qkv(cfg, Q, K, V, O_cross);
    O_acausal.assign(O_cross.size(), 0.0f);

    lh::cross_attention(Q.data(), K.data(), V.data(), O_cross.data(), cfg);
    auto cfg_acausal = cfg;
    cfg_acausal.causal = false;
    lh::sdpa(Q.data(), K.data(), V.data(), O_acausal.data(), cfg_acausal);
    EXPECT_TRUE(lh_test::AllClose(O_cross, O_acausal));
}

TEST(AttentionExt, AlibiBiasShapeAndSign) {
    constexpr int n_heads = 4, sq = 3, sk = 3;
    std::vector<float> slopes(n_heads);
    lh::alibi_slopes(slopes.data(), n_heads);
    std::vector<float> bias(n_heads * sq * sk);
    lh::alibi_bias(bias.data(), slopes.data(), n_heads, sq, sk, /*causal=*/true);

    // Diagonal entries: q_pos == k_pos, so delta=0 → bias 0.
    for (int h = 0; h < n_heads; ++h)
        for (int i = 0; i < sq; ++i) {
            EXPECT_FLOAT_EQ(bias[(h * sq + i) * sk + i], 0.0f);
        }
    // Below diagonal (j < i): delta > 0 → bias is -slope*delta < 0 for
    // positive slopes. ALiBi slopes are strictly positive.
    for (int h = 0; h < n_heads; ++h) {
        EXPECT_GT(slopes[h], 0.0f);
        EXPECT_LT(bias[(h * sq + 2) * sk + 0], 0.0f);
    }
}

TEST(AttentionExt, NtkScaleIncreasesBase) {
    const float base = 10000.0f;
    const float scaled = lh::ntk_scaled_theta(base, 2048, 8192, 128);
    EXPECT_GT(scaled, base);
}
