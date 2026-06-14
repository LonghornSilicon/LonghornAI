// LonghornAI — prefix cache (radix tree of token prefixes).
//
// Maintains a tree of token-id prefixes whose interior nodes carry the
// physical KV blocks that hold the K/V for those tokens. Used by the
// scheduler / cache manager to bit-exactly reuse blocks across requests
// that share a prefix (system prompts, common templates, retrieved
// documents).
//
// Granularity: only block-aligned prefixes are recorded. A prefix of
// length L*block_size produces L physical blocks; any tokens past the last
// block boundary are not cached because their KV depends on tokens not yet
// committed.
//
// Refcounts: each cached block holds one refcount in `CacheManager`. A
// `lookup` that returns N blocks bumps each by 1; the caller must
// `CacheManager::release_block` them when the request that referenced them
// completes (or carry them through the request's `block_table`, which has
// the same effect via `release_request`).
//
// Eviction: LRU over block-end nodes. `evict_until_free(k)` drops the
// least-recently-touched cached block(s) until the cache manager has at
// least `k` free physical blocks.
#ifndef LONGHORNAI_RUNTIME_PREFIX_CACHE_HPP
#define LONGHORNAI_RUNTIME_PREFIX_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "runtime/cache_manager.hpp"

namespace lh {

class PrefixCache {
public:
    explicit PrefixCache(CacheManager& mgr);

    struct Match {
        int matched_tokens = 0;            // always a multiple of block_size
        std::vector<int32_t> blocks;       // physical block ids, refcounted
    };

    // Walk the tree matching token-by-token; return the longest block-
    // aligned prefix and its physical blocks. Each returned block has its
    // refcount bumped.
    Match lookup(const int32_t* token_ids, int n_tokens);

    // Record (token_ids, blocks) into the tree. `n_blocks` blocks cover
    // `n_blocks * block_size` tokens; any extra tokens in `token_ids`
    // beyond that are ignored. Each cached block gets one refcount bump
    // owned by the prefix cache itself.
    void insert(const int32_t* token_ids, const int32_t* blocks, int n_blocks);

    // LRU eviction. Drops the least-recently-touched block-end nodes
    // (releasing their refcount and pruning empty branches) until the
    // cache manager has at least `min_free` free physical blocks. Returns
    // the number of blocks freed.
    int evict_until_free(int min_free);

    size_t num_cached_blocks() const { return cached_block_count_; }
    int block_size() const { return block_size_; }

private:
    struct Node {
        int32_t token_id = 0;
        int32_t physical_block = kInvalidBlock;
        Node* parent = nullptr;
        std::unordered_map<int32_t, std::unique_ptr<Node>> children;
        uint64_t access_tick = 0;
    };

    Node* descend_or_match(const int32_t* token_ids, int n_tokens) const;
    void prune_empty(Node* node);

    CacheManager* mgr_;
    int block_size_;
    Node root_;
    uint64_t tick_ = 0;
    size_t cached_block_count_ = 0;
};

}  // namespace lh

#endif  // LONGHORNAI_RUNTIME_PREFIX_CACHE_HPP
