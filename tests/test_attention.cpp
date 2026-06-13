#include <gtest/gtest.h>

#include <vector>

#include "kernels/attention.hpp"
#include "test_util.hpp"

namespace {
lh::AttnConfig make_cfg(int batch, int qh, int kvh, int sq, int sk, int d,
                        bool causal) {
    lh::AttnConfig c;
    c.batch = batch;
    c.n_q_heads = qh;
    c.n_kv_heads = kvh;
    c.seq_q = sq;
    c.seq_k = sk;
    c.head_dim = d;
    c.causal = causal;
    return c;
}

void run_pair(const lh::AttnConfig& cfg, uint32_t seed) {
    const int64_t qn =
        int64_t(cfg.batch) * cfg.n_q_heads * cfg.seq_q * cfg.head_dim;
    const int64_t kn =
        int64_t(cfg.batch) * cfg.n_kv_heads * cfg.seq_k * cfg.head_dim;
    auto Q = lh_test::random_vector(qn, seed);
    auto K = lh_test::random_vector(kn, seed + 1);
    auto V = lh_test::random_vector(kn, seed + 2);
    std::vector<float> O_sdpa(qn), O_flash(qn);

    lh::sdpa(Q.data(), K.data(), V.data(), O_sdpa.data(), cfg);
    lh::flash_attention(Q.data(), K.data(), V.data(), O_flash.data(), cfg, 16);
    EXPECT_TRUE(lh_test::AllClose(O_flash, O_sdpa, 1e-4f, 1e-4f));
}
}  // namespace

TEST(Attention, FlashMatchesSdpaMHA) {
    run_pair(make_cfg(2, 4, 4, 16, 16, 32, false), 700);
}

TEST(Attention, FlashMatchesSdpaCausal) {
    run_pair(make_cfg(1, 4, 4, 24, 24, 40, true), 710);
}

TEST(Attention, FlashMatchesSdpaMQA) {
    run_pair(make_cfg(1, 8, 1, 12, 12, 32, true), 720);
}

TEST(Attention, FlashMatchesSdpaGQA) {
    run_pair(make_cfg(2, 8, 2, 10, 10, 48, true), 730);
}

TEST(Attention, DecodeShapeSeqQLessThanSeqK) {
    // Single-token decode against a longer cached key sequence.
    run_pair(make_cfg(1, 4, 2, 1, 20, 32, true), 740);
}

TEST(Attention, KvCacheAppendEqualsFullRecompute) {
    // Build a sequence in two chunks via the cache, then attend; compare to
    // attending over the full K/V computed up front.
    const int n_kv = 2, d = 16, max_seq = 32;
    const int qh = 4, sq = 1;
    const int chunk1 = 10, chunk2 = 6, total = chunk1 + chunk2;

    auto Kfull = lh_test::random_vector(n_kv * total * d, 800);
    auto Vfull = lh_test::random_vector(n_kv * total * d, 801);

    // Split Kfull/Vfull (layout [n_kv, total, d]) into two time chunks.
    auto slice = [&](const std::vector<float>& src, int start, int len) {
        std::vector<float> out(n_kv * len * d);
        for (int h = 0; h < n_kv; ++h)
            for (int s = 0; s < len; ++s)
                for (int t = 0; t < d; ++t)
                    out[(h * len + s) * d + t] =
                        src[(h * total + (start + s)) * d + t];
        return out;
    };
    auto K1 = slice(Kfull, 0, chunk1), V1 = slice(Vfull, 0, chunk1);
    auto K2 = slice(Kfull, chunk1, chunk2), V2 = slice(Vfull, chunk1, chunk2);

    std::vector<float> kc(n_kv * max_seq * d, 0.0f), vc(n_kv * max_seq * d, 0.0f);
    lh::kv_cache_append(kc.data(), vc.data(), K1.data(), V1.data(), n_kv, d,
                        max_seq, 0, chunk1);
    lh::kv_cache_append(kc.data(), vc.data(), K2.data(), V2.data(), n_kv, d,
                        max_seq, chunk1, chunk2);

    // Verify cache contents match the contiguous full sequence.
    for (int h = 0; h < n_kv; ++h)
        for (int s = 0; s < total; ++s)
            for (int t = 0; t < d; ++t) {
                EXPECT_FLOAT_EQ(kc[(h * max_seq + s) * d + t],
                                Kfull[(h * total + s) * d + t]);
                EXPECT_FLOAT_EQ(vc[(h * max_seq + s) * d + t],
                                Vfull[(h * total + s) * d + t]);
            }

    auto Q = lh_test::random_vector(qh * sq * d, 802);
    auto cfg_cache = make_cfg(1, qh, n_kv, sq, total, d, true);
    cfg_cache.seq_k = total;

    // Attention reads contiguous [n_kv, total, d] from the cache front.
    std::vector<float> kc_view(n_kv * total * d), vc_view(n_kv * total * d);
    for (int h = 0; h < n_kv; ++h)
        for (int s = 0; s < total; ++s)
            for (int t = 0; t < d; ++t) {
                kc_view[(h * total + s) * d + t] = kc[(h * max_seq + s) * d + t];
                vc_view[(h * total + s) * d + t] = vc[(h * max_seq + s) * d + t];
            }

    std::vector<float> O_cache(qh * sq * d), O_full(qh * sq * d);
    lh::sdpa(Q.data(), kc_view.data(), vc_view.data(), O_cache.data(),
             cfg_cache);
    lh::sdpa(Q.data(), Kfull.data(), Vfull.data(), O_full.data(), cfg_cache);
    EXPECT_TRUE(lh_test::AllClose(O_cache, O_full));
}
