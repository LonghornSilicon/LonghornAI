#include <gtest/gtest.h>

#include <vector>

#include "kernels/attention.hpp"
#include "kernels/kvcache.hpp"
#include "kernels/paged_attention.hpp"
#include "runtime/cache_manager.hpp"
#include "test_util.hpp"

namespace {

lh::PagedCacheLayout small_layout(int num_blocks = 8, int block_size = 4,
                                  int n_kv = 2, int d = 8) {
    lh::PagedCacheLayout L;
    L.num_blocks = num_blocks;
    L.block_size = block_size;
    L.n_kv_heads = n_kv;
    L.head_dim = d;
    return L;
}

}  // namespace

TEST(CacheManager, AllocateAndReleaseTracksFreeList) {
    lh::CacheManager mgr(small_layout(/*num_blocks=*/4));
    EXPECT_EQ(mgr.num_free_blocks(), 4);
    int32_t a = mgr.allocate_block();
    int32_t b = mgr.allocate_block();
    EXPECT_NE(a, lh::kInvalidBlock);
    EXPECT_NE(b, lh::kInvalidBlock);
    EXPECT_EQ(mgr.num_free_blocks(), 2);
    EXPECT_EQ(mgr.refcount(a), 1);
    EXPECT_EQ(mgr.refcount(b), 1);
    mgr.release_block(a);
    EXPECT_EQ(mgr.num_free_blocks(), 3);
    EXPECT_EQ(mgr.refcount(a), 0);
}

TEST(CacheManager, OutOfMemoryReturnsSentinel) {
    lh::CacheManager mgr(small_layout(/*num_blocks=*/2));
    EXPECT_NE(mgr.allocate_block(), lh::kInvalidBlock);
    EXPECT_NE(mgr.allocate_block(), lh::kInvalidBlock);
    EXPECT_EQ(mgr.allocate_block(), lh::kInvalidBlock);
    EXPECT_EQ(mgr.num_free_blocks(), 0);
}

TEST(CacheManager, EnsureCapacityIsAllOrNothing) {
    lh::CacheManager mgr(small_layout(/*num_blocks=*/3, /*block_size=*/4));
    auto rid = mgr.create_request();
    // 3 blocks total. Asking for 13 tokens needs 4 blocks → OOM, table
    // must remain empty.
    EXPECT_FALSE(mgr.ensure_capacity(rid, 13));
    EXPECT_TRUE(mgr.blocks(rid).block_table.empty());
    // 9 tokens needs 3 blocks → OK.
    EXPECT_TRUE(mgr.ensure_capacity(rid, 9));
    EXPECT_EQ(mgr.blocks(rid).block_table.size(), 3u);
    EXPECT_EQ(mgr.num_free_blocks(), 0);
}

TEST(CacheManager, ReleaseRequestDropsBlocks) {
    lh::CacheManager mgr(small_layout(/*num_blocks=*/4, /*block_size=*/4));
    auto r1 = mgr.create_request();
    auto r2 = mgr.create_request();
    EXPECT_TRUE(mgr.ensure_capacity(r1, 8));   // 2 blocks
    EXPECT_TRUE(mgr.ensure_capacity(r2, 8));   // 2 blocks
    EXPECT_EQ(mgr.num_free_blocks(), 0);
    mgr.release_request(r1);
    EXPECT_EQ(mgr.num_free_blocks(), 2);
    mgr.release_request(r2);
    EXPECT_EQ(mgr.num_free_blocks(), 4);
}

TEST(CacheManager, SharedBlockPersistsThroughRefcount) {
    lh::CacheManager mgr(small_layout(/*num_blocks=*/4, /*block_size=*/4));
    auto r1 = mgr.create_request();
    auto r2 = mgr.create_request();
    EXPECT_TRUE(mgr.ensure_capacity(r1, 4));   // 1 block
    const int32_t shared = mgr.blocks(r1).block_table[0];
    EXPECT_EQ(mgr.refcount(shared), 1);

    // r2 references the same physical block at logical index 0.
    EXPECT_TRUE(mgr.reference_block(r2, 0, shared));
    EXPECT_EQ(mgr.refcount(shared), 2);

    // Releasing r1 keeps the block alive (still held by r2).
    mgr.release_request(r1);
    EXPECT_EQ(mgr.refcount(shared), 1);
    EXPECT_EQ(mgr.num_free_blocks(), 3);

    mgr.release_request(r2);
    EXPECT_EQ(mgr.refcount(shared), 0);
    EXPECT_EQ(mgr.num_free_blocks(), 4);
}

TEST(CacheManager, EndToEndAttendsThroughManagedBlocks) {
    // Walk a complete: create request → allocate blocks → append KV →
    // paged_attention → compare to SDPA.
    constexpr int qh = 2, kvh = 2, sq = 1, sk = 11, d = 16;
    constexpr int block_size = 4;

    auto Q = lh_test::random_vector(qh * sq * d, 1400);
    auto K = lh_test::random_vector(kvh * sk * d, 1401);
    auto V = lh_test::random_vector(kvh * sk * d, 1402);

    lh::PagedCacheLayout L;
    L.num_blocks = 8;
    L.block_size = block_size;
    L.n_kv_heads = kvh;
    L.head_dim = d;
    lh::CacheManager mgr(L);

    auto rid = mgr.create_request();
    EXPECT_TRUE(mgr.ensure_capacity(rid, sk));
    mgr.set_seq_len(rid, sk);
    const auto& tbl = mgr.blocks(rid).block_table;
    lh::paged_kv_append(mgr.k_pool(), mgr.v_pool(), K.data(), V.data(),
                        tbl.data(), L, 0, sk);

    lh::PagedAttnConfig pcfg;
    pcfg.n_q_heads = qh;
    pcfg.n_kv_heads = kvh;
    pcfg.seq_q = sq;
    pcfg.head_dim = d;
    pcfg.causal = true;
    std::vector<float> O_paged(qh * sq * d, 0.0f);
    lh::paged_attention(Q.data(), mgr.k_pool(), mgr.v_pool(), tbl.data(),
                        mgr.blocks(rid).seq_len, O_paged.data(), pcfg, L);

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
    EXPECT_TRUE(lh_test::AllClose(O_paged, O_sdpa, 1e-4f, 1e-4f));
}
