// LonghornAI — KV cache manager.
//
// Owns the physical block pool defined by `PagedCacheLayout` and manages
// per-request `block_table`s. Tracks per-block reference counts so prefix-
// shared blocks survive across requests; a block is returned to the free
// list only when its refcount drops to zero.
//
// Allocation policy is a free-list (LIFO) over physical block ids. The
// free list starts with all `num_blocks` ids; allocations pop, frees push.
// Out-of-memory returns a null sentinel so callers can decide whether to
// queue / evict / reject.
//
// Thread safety: not thread-safe by intent. The serving scheduler runs as
// a single producer; multi-threaded callers must serialise externally.
#ifndef LONGHORNAI_RUNTIME_CACHE_MANAGER_HPP
#define LONGHORNAI_RUNTIME_CACHE_MANAGER_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kernels/kvcache.hpp"

namespace lh {

using RequestId = int32_t;
constexpr int32_t kInvalidBlock = -1;

struct RequestBlocks {
    std::vector<int32_t> block_table;  // logical -> physical
    int seq_len = 0;                   // valid token positions
};

class CacheManager {
public:
    explicit CacheManager(const PagedCacheLayout& layout);

    const PagedCacheLayout& layout() const { return layout_; }

    int num_free_blocks() const {
        return static_cast<int>(free_list_.size());
    }

    int num_total_blocks() const { return layout_.num_blocks; }

    // Create a fresh request with no allocated blocks. Returns the request
    // id; ids are dense small integers reused after `release_request`.
    RequestId create_request();

    // Release a request: drop refcounts on its blocks, free those that
    // reach zero, drop the request entry. Subsequent calls on the id are
    // undefined.
    void release_request(RequestId id);

    // Ensure the request has enough blocks to cover `seq_len` tokens.
    // Returns true on success, false if OOM (no partial allocation: on
    // failure the request's block table is unchanged).
    bool ensure_capacity(RequestId id, int seq_len);

    // Mark the request's logical sequence length. Used by the scheduler so
    // attention sees the correct `seq_len`.
    void set_seq_len(RequestId id, int seq_len);

    // Reference an existing physical block at logical position
    // `logical_idx` of the request. Bumps the block's refcount; used by
    // the prefix cache to share blocks across requests.
    bool reference_block(RequestId id, int logical_idx, int32_t physical);

    const RequestBlocks& blocks(RequestId id) const {
        return requests_.at(id);
    }
    RequestBlocks& blocks_mut(RequestId id) { return requests_.at(id); }

    // Direct pool pointers; the scheduler/kernel calls hand these to
    // `paged_attention*` and `paged_kv_append`.
    float* k_pool() { return k_pool_.data(); }
    float* v_pool() { return v_pool_.data(); }
    const float* k_pool() const { return k_pool_.data(); }
    const float* v_pool() const { return v_pool_.data(); }

    // Per-block refcount inspection — exposed for tests and for the prefix
    // cache, which holds a refcount on each cached block.
    int refcount(int32_t physical) const {
        if (physical < 0 || physical >= layout_.num_blocks) return 0;
        return refcount_[static_cast<size_t>(physical)];
    }

    // Allocate one fresh block (refcount 1) and return its physical id, or
    // `kInvalidBlock` if OOM. The prefix cache owns blocks via this API
    // (and later `release_block`) without an associated request.
    int32_t allocate_block();

    // Drop one refcount on a physical block. When refcount reaches 0 the
    // block goes back on the free list.
    void release_block(int32_t physical);

    // Bump refcount on a physical block by 1.
    void retain_block(int32_t physical);

private:
    PagedCacheLayout layout_;
    std::vector<float> k_pool_;
    std::vector<float> v_pool_;
    std::vector<int32_t> free_list_;          // physical ids
    std::vector<int> refcount_;               // size = num_blocks

    std::unordered_map<RequestId, RequestBlocks> requests_;
    std::vector<RequestId> recycled_ids_;
    RequestId next_id_ = 0;
};

}  // namespace lh

#endif  // LONGHORNAI_RUNTIME_CACHE_MANAGER_HPP
