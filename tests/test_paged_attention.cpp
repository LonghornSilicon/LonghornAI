#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/kvcache.hpp"
#include "kernels/paged_attention.hpp"
#include "test_util.hpp"

namespace {

// Build a paged pool from a contiguous K/V tensor of shape [n_kv, S, D] by
// chunking S into blocks of `block_size`, allocating physical blocks
// sequentially, and recording the logical->physical map. This is the
// "trivial" allocation pattern (no fragmentation, no sharing); the point of
// the test is just that the paged kernel produces the same output as SDPA
// over the contiguous source.
struct PagedPool {
    std::vector<float> K_pool;
    std::vector<float> V_pool;
    std::vector<int32_t> table;
    lh::PagedCacheLayout layout;
};

PagedPool build_pool(const std::vector<float>& K, const std::vector<float>& V,
                     int n_kv, int S, int D, int block_size,
                     int extra_blocks = 2) {
    const int needed_blocks = (S + block_size - 1) / block_size;
    PagedPool p;
    p.layout.num_blocks = needed_blocks + extra_blocks;
    p.layout.block_size = block_size;
    p.layout.n_kv_heads = n_kv;
    p.layout.head_dim = D;
    const int64_t pool_n =
        static_cast<int64_t>(p.layout.num_blocks) * n_kv * block_size * D;
    p.K_pool.assign(static_cast<size_t>(pool_n), 0.0f);
    p.V_pool.assign(static_cast<size_t>(pool_n), 0.0f);
    p.table.assign(static_cast<size_t>(needed_blocks), 0);

    // Allocate physical blocks in reverse order to make sure the kernel
    // honours `block_table` rather than assuming logical == physical.
    for (int i = 0; i < needed_blocks; ++i) {
        p.table[i] = needed_blocks - 1 - i;
    }

    // Append the K/V token-by-token via `paged_kv_append`.
    lh::paged_kv_append(p.K_pool.data(), p.V_pool.data(), K.data(), V.data(),
                        p.table.data(), p.layout, /*past_len=*/0,
                        /*seq_new=*/S);
    return p;
}

}  // namespace

TEST(Paged, AppendThenAttendMatchesSdpa) {
    constexpr int qh = 4, kvh = 2, sq = 3, sk = 20, d = 16;
    constexpr int block_size = 8;

    auto Q = lh_test::random_vector(qh * sq * d, 1100);
    auto K = lh_test::random_vector(kvh * sk * d, 1101);
    auto V = lh_test::random_vector(kvh * sk * d, 1102);

    auto pool = build_pool(K, V, kvh, sk, d, block_size);

    // SDPA reference over the contiguous K/V.
    lh::AttnConfig cfg;
    cfg.batch = 1;
    cfg.n_q_heads = qh;
    cfg.n_kv_heads = kvh;
    cfg.seq_q = sq;
    cfg.seq_k = sk;
    cfg.head_dim = d;
    cfg.causal = true;
    std::vector<float> O_sdpa(qh * sq * d, 0.0f);
    lh::sdpa(Q.data(), K.data(), V.data(), O_sdpa.data(), cfg);

    // Paged attention.
    lh::PagedAttnConfig pcfg;
    pcfg.n_q_heads = qh;
    pcfg.n_kv_heads = kvh;
    pcfg.seq_q = sq;
    pcfg.head_dim = d;
    pcfg.causal = true;
    std::vector<float> O_paged(qh * sq * d, 0.0f);
    lh::paged_attention(Q.data(), pool.K_pool.data(), pool.V_pool.data(),
                        pool.table.data(), /*seq_len=*/sk, O_paged.data(),
                        pcfg, pool.layout);

    EXPECT_TRUE(lh_test::AllClose(O_paged, O_sdpa, 1e-4f, 1e-4f));
}

TEST(Paged, IncrementalAppendMatchesSdpa) {
    // Append K/V in two chunks instead of one and verify the kernel still
    // matches SDPA over the full contiguous K/V.
    constexpr int kvh = 2, d = 16;
    constexpr int block_size = 8;
    constexpr int chunk1 = 13, chunk2 = 11, total = chunk1 + chunk2;
    constexpr int qh = 2, sq = 1;

    auto Kfull = lh_test::random_vector(kvh * total * d, 1200);
    auto Vfull = lh_test::random_vector(kvh * total * d, 1201);

    // Slice [n_kv, total, d] into time chunks.
    auto slice = [&](const std::vector<float>& src, int start, int len) {
        std::vector<float> out(kvh * len * d);
        for (int h = 0; h < kvh; ++h)
            for (int s = 0; s < len; ++s)
                for (int t = 0; t < d; ++t)
                    out[(h * len + s) * d + t] =
                        src[(h * total + (start + s)) * d + t];
        return out;
    };
    auto K1 = slice(Kfull, 0, chunk1);
    auto V1 = slice(Vfull, 0, chunk1);
    auto K2 = slice(Kfull, chunk1, chunk2);
    auto V2 = slice(Vfull, chunk1, chunk2);

    // Allocate a pool large enough for `total`.
    lh::PagedCacheLayout L;
    L.block_size = block_size;
    L.n_kv_heads = kvh;
    L.head_dim = d;
    const int needed_blocks = (total + block_size - 1) / block_size;
    L.num_blocks = needed_blocks + 1;
    std::vector<float> Kp(L.num_blocks * kvh * block_size * d, 0.0f);
    std::vector<float> Vp(L.num_blocks * kvh * block_size * d, 0.0f);

    // Block-table allocation in reverse order again (logical != physical).
    std::vector<int32_t> table(needed_blocks);
    for (int i = 0; i < needed_blocks; ++i) {
        table[i] = needed_blocks - 1 - i;
    }

    lh::paged_kv_append(Kp.data(), Vp.data(), K1.data(), V1.data(),
                        table.data(), L, /*past_len=*/0, chunk1);
    lh::paged_kv_append(Kp.data(), Vp.data(), K2.data(), V2.data(),
                        table.data(), L, /*past_len=*/chunk1, chunk2);

    auto Q = lh_test::random_vector(qh * sq * d, 1202);

    lh::AttnConfig cfg;
    cfg.batch = 1;
    cfg.n_q_heads = qh;
    cfg.n_kv_heads = kvh;
    cfg.seq_q = sq;
    cfg.seq_k = total;
    cfg.head_dim = d;
    cfg.causal = true;
    std::vector<float> O_sdpa(qh * sq * d, 0.0f);
    lh::sdpa(Q.data(), Kfull.data(), Vfull.data(), O_sdpa.data(), cfg);

    lh::PagedAttnConfig pcfg;
    pcfg.n_q_heads = qh;
    pcfg.n_kv_heads = kvh;
    pcfg.seq_q = sq;
    pcfg.head_dim = d;
    pcfg.causal = true;
    std::vector<float> O_paged(qh * sq * d, 0.0f);
    lh::paged_attention(Q.data(), Kp.data(), Vp.data(), table.data(), total,
                        O_paged.data(), pcfg, L);
    EXPECT_TRUE(lh_test::AllClose(O_paged, O_sdpa, 1e-4f, 1e-4f));
}

TEST(Paged, BatchedMatchesPerRequest) {
    // Two requests of different lengths, sharing one pool. Build them
    // independently and verify the batched kernel matches per-request runs.
    constexpr int qh = 2, kvh = 2, sq = 1, d = 16;
    constexpr int block_size = 8;

    struct Req {
        int sk;
        std::vector<float> Q, K, V;
        std::vector<int32_t> table_logical;
        int phys_offset;  // first physical block id for this request
    };

    std::vector<Req> reqs(2);
    reqs[0].sk = 17;
    reqs[1].sk = 9;
    reqs[0].Q = lh_test::random_vector(qh * sq * d, 1300);
    reqs[1].Q = lh_test::random_vector(qh * sq * d, 1310);
    reqs[0].K = lh_test::random_vector(kvh * reqs[0].sk * d, 1301);
    reqs[0].V = lh_test::random_vector(kvh * reqs[0].sk * d, 1302);
    reqs[1].K = lh_test::random_vector(kvh * reqs[1].sk * d, 1311);
    reqs[1].V = lh_test::random_vector(kvh * reqs[1].sk * d, 1312);

    // Pool sized for both requests with a few extra blocks.
    lh::PagedCacheLayout L;
    L.block_size = block_size;
    L.n_kv_heads = kvh;
    L.head_dim = d;
    const int b0 = (reqs[0].sk + block_size - 1) / block_size;
    const int b1 = (reqs[1].sk + block_size - 1) / block_size;
    L.num_blocks = b0 + b1 + 2;
    std::vector<float> Kp(L.num_blocks * kvh * block_size * d, 0.0f);
    std::vector<float> Vp(L.num_blocks * kvh * block_size * d, 0.0f);

    // Interleaved physical allocation: req0 gets blocks (0, 2, 4, ...),
    // req1 gets (1, 3, 5, ...). Forces the kernel to actually use the
    // per-request block table.
    reqs[0].table_logical.resize(b0);
    reqs[1].table_logical.resize(b1);
    int next = 0;
    for (int i = 0; i < std::max(b0, b1); ++i) {
        if (i < b0) reqs[0].table_logical[i] = next++;
        if (i < b1) reqs[1].table_logical[i] = next++;
    }
    reqs[0].phys_offset = reqs[0].table_logical[0];
    reqs[1].phys_offset = reqs[1].table_logical[0];

    lh::paged_kv_append(Kp.data(), Vp.data(), reqs[0].K.data(),
                        reqs[0].V.data(), reqs[0].table_logical.data(), L, 0,
                        reqs[0].sk);
    lh::paged_kv_append(Kp.data(), Vp.data(), reqs[1].K.data(),
                        reqs[1].V.data(), reqs[1].table_logical.data(), L, 0,
                        reqs[1].sk);

    lh::PagedAttnConfig pcfg;
    pcfg.n_q_heads = qh;
    pcfg.n_kv_heads = kvh;
    pcfg.seq_q = sq;
    pcfg.head_dim = d;
    pcfg.causal = true;

    // Per-request reference.
    std::vector<float> O0_solo(qh * sq * d, 0.0f);
    std::vector<float> O1_solo(qh * sq * d, 0.0f);
    lh::paged_attention(reqs[0].Q.data(), Kp.data(), Vp.data(),
                        reqs[0].table_logical.data(), reqs[0].sk,
                        O0_solo.data(), pcfg, L);
    lh::paged_attention(reqs[1].Q.data(), Kp.data(), Vp.data(),
                        reqs[1].table_logical.data(), reqs[1].sk,
                        O1_solo.data(), pcfg, L);

    // Batched run.
    const int max_blocks = std::max(b0, b1);
    std::vector<int32_t> block_tables(2 * max_blocks, 0);
    for (int i = 0; i < b0; ++i) block_tables[i] = reqs[0].table_logical[i];
    for (int i = 0; i < b1; ++i)
        block_tables[max_blocks + i] = reqs[1].table_logical[i];
    std::vector<int32_t> seq_lens = {reqs[0].sk, reqs[1].sk};

    std::vector<float> Q_b(2 * qh * sq * d, 0.0f);
    std::copy(reqs[0].Q.begin(), reqs[0].Q.end(), Q_b.begin());
    std::copy(reqs[1].Q.begin(), reqs[1].Q.end(), Q_b.begin() + qh * sq * d);
    std::vector<float> O_b(2 * qh * sq * d, 0.0f);
    lh::paged_attention_batched(Q_b.data(), Kp.data(), Vp.data(),
                                block_tables.data(), max_blocks,
                                seq_lens.data(), 2, O_b.data(), pcfg, L);

    std::vector<float> O0_batched(O_b.begin(), O_b.begin() + qh * sq * d);
    std::vector<float> O1_batched(O_b.begin() + qh * sq * d, O_b.end());
    EXPECT_TRUE(lh_test::AllClose(O0_batched, O0_solo));
    EXPECT_TRUE(lh_test::AllClose(O1_batched, O1_solo));
}
