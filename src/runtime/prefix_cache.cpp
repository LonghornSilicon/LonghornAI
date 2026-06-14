#include "runtime/prefix_cache.hpp"

#include <limits>

namespace lh {

PrefixCache::PrefixCache(CacheManager& mgr)
    : mgr_(&mgr), block_size_(mgr.layout().block_size) {}

PrefixCache::Match PrefixCache::lookup(const int32_t* token_ids,
                                       int n_tokens) {
    Match m;
    if (n_tokens < block_size_) return m;

    Node* cur = &root_;
    int matched = 0;

    for (int t = 0; t < n_tokens; ++t) {
        auto it = cur->children.find(token_ids[t]);
        if (it == cur->children.end()) break;
        cur = it->second.get();
        ++matched;
    }

    // Trim to a multiple of block_size and back up `cur` accordingly.
    const int blocks = matched / block_size_;
    if (blocks == 0) return m;
    const int trimmed = blocks * block_size_;
    while (matched > trimmed) {
        cur = cur->parent;
        --matched;
    }

    // Collect block ids deepest-first while bumping refcounts and LRU
    // ticks; reverse once at the end.
    m.matched_tokens = trimmed;
    m.blocks.reserve(static_cast<size_t>(blocks));
    ++tick_;
    Node* node = cur;
    while (node && node != &root_) {
        if (node->physical_block != kInvalidBlock) {
            m.blocks.push_back(node->physical_block);
            mgr_->retain_block(node->physical_block);
            node->access_tick = tick_;
        }
        node = node->parent;
    }
    for (size_t i = 0, j = m.blocks.size(); i + 1 < j; ++i, --j) {
        std::swap(m.blocks[i], m.blocks[j - 1]);
    }
    return m;
}

void PrefixCache::insert(const int32_t* token_ids, const int32_t* blocks,
                         int n_blocks) {
    if (n_blocks <= 0) return;
    Node* cur = &root_;
    ++tick_;
    for (int b = 0; b < n_blocks; ++b) {
        for (int t = 0; t < block_size_; ++t) {
            const int32_t tok = token_ids[b * block_size_ + t];
            auto& child = cur->children[tok];
            if (!child) {
                child = std::make_unique<Node>();
                child->token_id = tok;
                child->parent = cur;
            }
            cur = child.get();
        }
        // We are now at the block-end node. If it doesn't yet hold a
        // physical block, attach this one (refcount +1). If it already
        // does and matches, we don't double-cache; if it differs, we keep
        // the existing one (the prefix is canonical) — the new caller's
        // copy will be released when its request completes.
        if (cur->physical_block == kInvalidBlock) {
            cur->physical_block = blocks[b];
            mgr_->retain_block(blocks[b]);
            ++cached_block_count_;
        }
        cur->access_tick = tick_;
    }
}

namespace {

// Recursive scan of all block-end nodes; returns the one with the smallest
// `access_tick`. Templated on the node type so it can stay in an anonymous
// namespace despite `Node` being a private member type.
template <class N>
void find_lru(N* root, N*& victim, uint64_t& min_tick) {
    if (root->physical_block != kInvalidBlock) {
        if (root->access_tick < min_tick) {
            min_tick = root->access_tick;
            victim = root;
        }
    }
    for (auto& kv : root->children) find_lru(kv.second.get(), victim, min_tick);
}

}  // namespace

void PrefixCache::prune_empty(Node* node) {
    while (node && node != &root_ && node->children.empty() &&
           node->physical_block == kInvalidBlock) {
        Node* parent = node->parent;
        // Unlink: parent owns `node` via unique_ptr in its children map.
        parent->children.erase(node->token_id);
        node = parent;
    }
}

int PrefixCache::evict_until_free(int min_free) {
    int freed = 0;
    while (mgr_->num_free_blocks() < min_free) {
        Node* victim = nullptr;
        uint64_t min_tick = std::numeric_limits<uint64_t>::max();
        find_lru<Node>(&root_, victim, min_tick);
        if (!victim) break;  // nothing left to evict
        const int32_t blk = victim->physical_block;
        victim->physical_block = kInvalidBlock;
        if (cached_block_count_ > 0) --cached_block_count_;
        mgr_->release_block(blk);
        ++freed;
        prune_empty(victim);
    }
    return freed;
}

}  // namespace lh
