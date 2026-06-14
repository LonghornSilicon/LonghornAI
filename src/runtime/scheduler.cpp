#include "runtime/scheduler.hpp"

#include <algorithm>
#include <cassert>

namespace lh {

Scheduler::Scheduler(CacheManager& mgr, int max_concurrent)
    : mgr_(&mgr), max_concurrent_(max_concurrent) {}

RequestId Scheduler::submit(Request req) {
    const auto id = mgr_->create_request();
    queue_.emplace_back(id, std::move(req));
    return id;
}

bool Scheduler::try_admit_one(const Request& req, RequestId id) {
    // Reserve enough blocks for the prompt plus one decode slot.
    const int needed_tokens =
        static_cast<int>(req.prompt_tokens.size()) + 1;
    if (!mgr_->ensure_capacity(id, needed_tokens)) return false;
    mgr_->set_seq_len(id, static_cast<int>(req.prompt_tokens.size()));
    active_.emplace(id, ActiveState{req, /*produced=*/0});
    active_order_.push_back(id);
    return true;
}

ActiveStep Scheduler::prepare_step() {
    // Admission. Iterate the queue head while there's room (in both the
    // active set and the KV pool). On OOM we stop trying — admitting a
    // shorter request behind a long-blocked one would violate FIFO and
    // also wouldn't help unblock the head request.
    while (!queue_.empty() &&
           static_cast<int>(active_.size()) < max_concurrent_) {
        auto& [id, req] = queue_.front();
        if (!try_admit_one(req, id)) break;
        queue_.pop_front();
    }

    // Build the step view from the current active set.
    ActiveStep s;
    int max_blocks = 0;
    for (RequestId id : active_order_) {
        const auto& blocks = mgr_->blocks(id);
        max_blocks =
            std::max(max_blocks, static_cast<int>(blocks.block_table.size()));
    }
    s.max_blocks_per_req = max_blocks;
    s.ids = active_order_;
    s.seq_lens.reserve(active_order_.size());
    s.block_tables.assign(active_order_.size() * max_blocks, kInvalidBlock);
    for (size_t i = 0; i < active_order_.size(); ++i) {
        const auto& blocks = mgr_->blocks(active_order_[i]);
        s.seq_lens.push_back(blocks.seq_len);
        for (size_t b = 0; b < blocks.block_table.size(); ++b) {
            s.block_tables[i * max_blocks + b] = blocks.block_table[b];
        }
    }

    last_occ_ = max_concurrent_ > 0
                    ? static_cast<double>(active_.size()) / max_concurrent_
                    : 0.0;
    return s;
}

void Scheduler::commit_step(const std::vector<bool>& completed,
                            const std::vector<int32_t>& tokens) {
    assert(completed.size() == active_order_.size());
    (void)tokens;  // unused at this layer; sampling lives elsewhere
    std::vector<RequestId> survivors;
    survivors.reserve(active_order_.size());

    for (size_t i = 0; i < active_order_.size(); ++i) {
        const RequestId id = active_order_[i];
        auto it = active_.find(id);
        if (it == active_.end()) continue;
        auto& st = it->second;
        ++st.produced;
        const bool max_hit = st.produced >= st.req.max_new_tokens;
        if (completed[i] || max_hit) {
            generated_[id] = st.produced;
            mgr_->release_request(id);
            active_.erase(it);
            continue;
        }
        // Continuing: bump cache seq_len and ensure room for the next
        // decode slot.
        const int new_len = mgr_->blocks(id).seq_len + 1;
        // OOM here is rare (we ensured +1 at admission and on each step);
        // if it does happen, we mark the request completed early so the
        // caller can re-queue if desired.
        if (!mgr_->ensure_capacity(id, new_len + 1)) {
            generated_[id] = st.produced;
            mgr_->release_request(id);
            active_.erase(it);
            continue;
        }
        mgr_->set_seq_len(id, new_len);
        survivors.push_back(id);
    }
    active_order_.swap(survivors);
}

}  // namespace lh
