// LonghornAI — continuous-batching scheduler.
//
// Manages the request lifecycle for a paged-KV inference engine: queueing,
// admission, step batching, and completion. Acts as the bridge between
// per-request `Request` objects and the kernel layer's batched
// `paged_attention*` calls.
//
// The scheduler does not run the model itself. Its API is designed so the
// caller drives the inference loop:
//
//   while (sched.has_work()) {
//     auto view = sched.prepare_step();   // current active set + tables
//     // (caller) build Q, run the model, decode one token per active req
//     sched.commit_step(completed_flags, next_tokens);
//   }
//
// Admission is FIFO and gated by KV-block availability. A queued request
// is admitted only if the cache manager can satisfy its full prompt plus
// at least one decode slot; on OOM it stays queued (no partial admission).
// `max_concurrent` further caps the active-set size for latency control.
#ifndef LONGHORNAI_RUNTIME_SCHEDULER_HPP
#define LONGHORNAI_RUNTIME_SCHEDULER_HPP

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#include "runtime/cache_manager.hpp"

namespace lh {

struct Request {
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens = 0;
};

struct ActiveStep {
    std::vector<RequestId> ids;
    std::vector<int32_t> seq_lens;
    std::vector<int32_t> block_tables;   // [n_active * max_blocks_per_req]
    int max_blocks_per_req = 0;

    int n_active() const { return static_cast<int>(ids.size()); }
};

class Scheduler {
public:
    Scheduler(CacheManager& mgr, int max_concurrent);

    int max_concurrent() const { return max_concurrent_; }

    RequestId submit(Request req);

    bool has_work() const {
        return !queue_.empty() || !active_.empty();
    }

    size_t queue_depth() const { return queue_.size(); }
    size_t active_size() const { return active_.size(); }

    // Last `prepare_step()`'s active count divided by `max_concurrent`.
    // 1.0 means the scheduler ran the kernel at full concurrency.
    double last_step_occupancy() const { return last_occ_; }

    // Per-request total decoded tokens (excludes prompt). Updated by
    // `commit_step`; used by the tests to verify request-level progress.
    int generated(RequestId id) const {
        auto it = generated_.find(id);
        return it == generated_.end() ? 0 : it->second;
    }

    // Build the current active step. Admits queued requests subject to KV
    // and concurrency limits; allocates blocks for newly-admitted prompts;
    // returns the snapshot the caller hands to `paged_attention_batched`.
    ActiveStep prepare_step();

    // Commit per-active decisions in `prepare_step()` order. `completed`
    // marks requests that produced EOS this step; `tokens` is unused in
    // this scheduler-only version (kept for symmetry with future sampling
    // integration). Both vectors must be of length `step.n_active()`.
    void commit_step(const std::vector<bool>& completed,
                     const std::vector<int32_t>& tokens);

private:
    // Try to admit one queued request. Returns true if admitted.
    bool try_admit_one(const Request& req, RequestId id);

    CacheManager* mgr_;
    int max_concurrent_;

    // Queue of (id, request body) waiting for admission.
    std::deque<std::pair<RequestId, Request>> queue_;

    struct ActiveState {
        Request req;          // copy of the original request body
        int produced = 0;     // tokens decoded so far (excludes prompt)
    };
    std::unordered_map<RequestId, ActiveState> active_;
    std::vector<RequestId> active_order_;  // order returned by prepare_step

    std::unordered_map<RequestId, int> generated_;  // post-completion counts
    double last_occ_ = 0.0;
};

}  // namespace lh

#endif  // LONGHORNAI_RUNTIME_SCHEDULER_HPP
