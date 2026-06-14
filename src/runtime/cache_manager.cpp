#include "runtime/cache_manager.hpp"

#include <algorithm>
#include <cassert>

namespace lh {

CacheManager::CacheManager(const PagedCacheLayout& layout) : layout_(layout) {
    const int64_t pool_n = static_cast<int64_t>(layout.num_blocks) *
                           layout.n_kv_heads * layout.block_size *
                           layout.head_dim;
    k_pool_.assign(static_cast<size_t>(pool_n), 0.0f);
    v_pool_.assign(static_cast<size_t>(pool_n), 0.0f);
    refcount_.assign(static_cast<size_t>(layout.num_blocks), 0);
    free_list_.reserve(static_cast<size_t>(layout.num_blocks));
    // Push in reverse so popping yields ascending physical ids — easier to
    // read in tests and traces.
    for (int i = layout.num_blocks - 1; i >= 0; --i) free_list_.push_back(i);
}

int32_t CacheManager::allocate_block() {
    if (free_list_.empty()) return kInvalidBlock;
    const int32_t b = free_list_.back();
    free_list_.pop_back();
    refcount_[static_cast<size_t>(b)] = 1;
    return b;
}

void CacheManager::release_block(int32_t physical) {
    if (physical < 0 || physical >= layout_.num_blocks) return;
    int& rc = refcount_[static_cast<size_t>(physical)];
    if (rc <= 0) return;  // already free; ignore double-release
    if (--rc == 0) free_list_.push_back(physical);
}

void CacheManager::retain_block(int32_t physical) {
    if (physical < 0 || physical >= layout_.num_blocks) return;
    ++refcount_[static_cast<size_t>(physical)];
}

RequestId CacheManager::create_request() {
    RequestId id;
    if (!recycled_ids_.empty()) {
        id = recycled_ids_.back();
        recycled_ids_.pop_back();
    } else {
        id = next_id_++;
    }
    requests_.emplace(id, RequestBlocks{});
    return id;
}

void CacheManager::release_request(RequestId id) {
    auto it = requests_.find(id);
    if (it == requests_.end()) return;
    for (int32_t b : it->second.block_table) release_block(b);
    requests_.erase(it);
    recycled_ids_.push_back(id);
}

bool CacheManager::ensure_capacity(RequestId id, int seq_len) {
    auto it = requests_.find(id);
    assert(it != requests_.end());
    auto& req = it->second;
    const int needed = (seq_len + layout_.block_size - 1) / layout_.block_size;
    const int have = static_cast<int>(req.block_table.size());
    if (needed <= have) return true;
    const int extra = needed - have;
    if (extra > num_free_blocks()) return false;  // OOM, no partial alloc
    req.block_table.reserve(static_cast<size_t>(needed));
    for (int i = 0; i < extra; ++i) {
        req.block_table.push_back(allocate_block());
    }
    return true;
}

void CacheManager::set_seq_len(RequestId id, int seq_len) {
    auto it = requests_.find(id);
    assert(it != requests_.end());
    it->second.seq_len = seq_len;
}

bool CacheManager::reference_block(RequestId id, int logical_idx,
                                   int32_t physical) {
    auto it = requests_.find(id);
    assert(it != requests_.end());
    auto& tbl = it->second.block_table;
    if (logical_idx < 0) return false;
    // Existing entry at `logical_idx` is replaced; release the old one.
    if (logical_idx < static_cast<int>(tbl.size())) {
        const int32_t old = tbl[static_cast<size_t>(logical_idx)];
        if (old >= 0) release_block(old);
        retain_block(physical);
        tbl[static_cast<size_t>(logical_idx)] = physical;
        return true;
    }
    // Pad with sentinels; caller should fill them in subsequent calls.
    tbl.resize(static_cast<size_t>(logical_idx + 1), kInvalidBlock);
    retain_block(physical);
    tbl[static_cast<size_t>(logical_idx)] = physical;
    return true;
}

}  // namespace lh
