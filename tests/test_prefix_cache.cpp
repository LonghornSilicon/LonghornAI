#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "kernels/kvcache.hpp"
#include "runtime/cache_manager.hpp"
#include "runtime/prefix_cache.hpp"
#include "test_util.hpp"

namespace {

lh::PagedCacheLayout L(int num_blocks = 8, int block_size = 4) {
    lh::PagedCacheLayout l;
    l.num_blocks = num_blocks;
    l.block_size = block_size;
    l.n_kv_heads = 1;
    l.head_dim = 4;
    return l;
}

// Helper: insert a "fake" prefix by allocating blocks via the manager and
// pushing them through the prefix cache. We don't write actual K/V here;
// only the bookkeeping is under test.
std::vector<int32_t> insert_prefix(lh::CacheManager& mgr,
                                   lh::PrefixCache& pc,
                                   const std::vector<int32_t>& token_ids) {
    const int B = pc.block_size();
    const int n_blocks = static_cast<int>(token_ids.size()) / B;
    std::vector<int32_t> blocks;
    blocks.reserve(static_cast<size_t>(n_blocks));
    for (int i = 0; i < n_blocks; ++i) {
        const int32_t b = mgr.allocate_block();
        EXPECT_NE(b, lh::kInvalidBlock);
        blocks.push_back(b);
    }
    pc.insert(token_ids.data(), blocks.data(), n_blocks);
    // Insert bumped refcount; release our caller-side refcount so only
    // the cache's hold remains.
    for (int32_t b : blocks) mgr.release_block(b);
    return blocks;
}

}  // namespace

TEST(PrefixCache, LookupWithoutInsertReturnsEmpty) {
    lh::CacheManager mgr(L());
    lh::PrefixCache pc(mgr);
    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    auto m = pc.lookup(ids.data(), static_cast<int>(ids.size()));
    EXPECT_EQ(m.matched_tokens, 0);
    EXPECT_TRUE(m.blocks.empty());
}

TEST(PrefixCache, ExactMatchReturnsBlocks) {
    lh::CacheManager mgr(L(/*num_blocks=*/8, /*block_size=*/4));
    lh::PrefixCache pc(mgr);
    std::vector<int32_t> seq = {10, 20, 30, 40, 50, 60, 70, 80};
    auto blocks = insert_prefix(mgr, pc, seq);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(pc.num_cached_blocks(), 2u);
    // After insert, cache holds both blocks at refcount 1.
    EXPECT_EQ(mgr.refcount(blocks[0]), 1);
    EXPECT_EQ(mgr.refcount(blocks[1]), 1);

    auto m = pc.lookup(seq.data(), static_cast<int>(seq.size()));
    EXPECT_EQ(m.matched_tokens, 8);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0], blocks[0]);
    EXPECT_EQ(m.blocks[1], blocks[1]);
    // Lookup bumped refcounts (we still hold them).
    EXPECT_EQ(mgr.refcount(blocks[0]), 2);
    EXPECT_EQ(mgr.refcount(blocks[1]), 2);

    // Release the lookup's holds.
    for (int32_t b : m.blocks) mgr.release_block(b);
    EXPECT_EQ(mgr.refcount(blocks[0]), 1);
}

TEST(PrefixCache, PartialMatchTruncatesToBlockBoundary) {
    lh::CacheManager mgr(L(/*num_blocks=*/8, /*block_size=*/4));
    lh::PrefixCache pc(mgr);
    std::vector<int32_t> seq = {10, 20, 30, 40, 50, 60, 70, 80};
    insert_prefix(mgr, pc, seq);

    // Lookup with a sequence that diverges mid-block-2: matches all of
    // block-0 (4 tokens), but block-1 is broken at position 5.
    std::vector<int32_t> probe = {10, 20, 30, 40, 50, 99, 99, 99};
    auto m = pc.lookup(probe.data(), static_cast<int>(probe.size()));
    EXPECT_EQ(m.matched_tokens, 4);
    EXPECT_EQ(m.blocks.size(), 1u);

    for (int32_t b : m.blocks) mgr.release_block(b);
}

TEST(PrefixCache, EvictDropsLeastRecentlyUsed) {
    lh::CacheManager mgr(L(/*num_blocks=*/4, /*block_size=*/4));
    lh::PrefixCache pc(mgr);

    // Two distinct prefixes, each one block. Pool has 4 blocks total.
    std::vector<int32_t> seq_a = {1, 2, 3, 4};
    std::vector<int32_t> seq_b = {9, 9, 9, 9};
    insert_prefix(mgr, pc, seq_a);
    insert_prefix(mgr, pc, seq_b);
    EXPECT_EQ(pc.num_cached_blocks(), 2u);
    EXPECT_EQ(mgr.num_free_blocks(), 2);

    // Touch seq_a so seq_b is the LRU.
    auto m = pc.lookup(seq_a.data(), static_cast<int>(seq_a.size()));
    for (int32_t b : m.blocks) mgr.release_block(b);

    // Demand 3 free blocks. Forces eviction of the LRU (seq_b).
    const int freed = pc.evict_until_free(3);
    EXPECT_EQ(freed, 1);
    EXPECT_EQ(mgr.num_free_blocks(), 3);

    // seq_a is still cached and still findable.
    auto m2 = pc.lookup(seq_a.data(), static_cast<int>(seq_a.size()));
    EXPECT_EQ(m2.matched_tokens, 4);
    for (int32_t b : m2.blocks) mgr.release_block(b);

    // seq_b is no longer cached.
    auto m3 = pc.lookup(seq_b.data(), static_cast<int>(seq_b.size()));
    EXPECT_EQ(m3.matched_tokens, 0);
}

TEST(PrefixCache, SharedRootSavesBlocksForOverlappingPrefixes) {
    // Two requests share the first block (4 tokens) but diverge in the
    // second. The cache should hold 3 blocks total (1 shared + 2
    // divergent).
    lh::CacheManager mgr(L(/*num_blocks=*/8, /*block_size=*/4));
    lh::PrefixCache pc(mgr);

    std::vector<int32_t> seq_a = {1, 2, 3, 4, 100, 200, 300, 400};
    std::vector<int32_t> seq_b = {1, 2, 3, 4, 500, 600, 700, 800};
    insert_prefix(mgr, pc, seq_a);
    // For seq_b, we already have a shared first block; we must reuse that
    // physical id for the prefix to be canonical. Look it up first, then
    // allocate only the divergent tail.
    auto matched = pc.lookup(seq_b.data(), static_cast<int>(seq_b.size()));
    EXPECT_EQ(matched.matched_tokens, 4);
    EXPECT_EQ(matched.blocks.size(), 1u);
    // Allocate one new block for the divergent tail.
    int32_t tail = mgr.allocate_block();
    ASSERT_NE(tail, lh::kInvalidBlock);
    std::vector<int32_t> blocks_b = {matched.blocks[0], tail};
    pc.insert(seq_b.data(), blocks_b.data(), 2);
    // Drop our caller-side refcounts (insert took its own).
    for (int32_t b : matched.blocks) mgr.release_block(b);
    mgr.release_block(tail);

    EXPECT_EQ(pc.num_cached_blocks(), 3u);
}
