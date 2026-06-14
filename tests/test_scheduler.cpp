#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "kernels/kvcache.hpp"
#include "runtime/cache_manager.hpp"
#include "runtime/scheduler.hpp"
#include "test_util.hpp"

namespace {

lh::PagedCacheLayout layout(int blocks, int block_size = 4) {
    lh::PagedCacheLayout L;
    L.num_blocks = blocks;
    L.block_size = block_size;
    L.n_kv_heads = 1;
    L.head_dim = 4;
    return L;
}

lh::Request make_request(int prompt_len, int max_new_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> tok(0, 31);
    lh::Request r;
    r.prompt_tokens.resize(static_cast<size_t>(prompt_len));
    for (auto& t : r.prompt_tokens) t = tok(rng);
    r.max_new_tokens = max_new_tokens;
    return r;
}

}  // namespace

TEST(Scheduler, AdmitsUpToMaxConcurrent) {
    lh::CacheManager mgr(layout(/*blocks=*/16));
    lh::Scheduler sched(mgr, /*max_concurrent=*/4);
    for (int i = 0; i < 6; ++i) {
        sched.submit(make_request(/*prompt=*/4, /*max_new=*/2,
                                  /*seed=*/static_cast<uint32_t>(i)));
    }
    auto step = sched.prepare_step();
    EXPECT_EQ(step.n_active(), 4);
    EXPECT_EQ(sched.queue_depth(), 2u);
    EXPECT_NEAR(sched.last_step_occupancy(), 1.0, 1e-9);
}

TEST(Scheduler, AdmissionStallsOnOOM) {
    // Pool has only 4 blocks (block_size = 4 → 16 total tokens of room).
    // Two requests of 12 prompt tokens each need 4 blocks (12 + 1 → 4
    // blocks = 16 tokens). Only one can be admitted; the second waits.
    lh::CacheManager mgr(layout(/*blocks=*/4));
    lh::Scheduler sched(mgr, /*max_concurrent=*/4);
    sched.submit(make_request(/*prompt=*/12, /*max_new=*/1, 100));
    sched.submit(make_request(/*prompt=*/12, /*max_new=*/1, 101));
    auto step = sched.prepare_step();
    EXPECT_EQ(step.n_active(), 1);
    EXPECT_EQ(sched.queue_depth(), 1u);
}

TEST(Scheduler, CompletedRequestFreesBlocks) {
    lh::CacheManager mgr(layout(/*blocks=*/8));
    lh::Scheduler sched(mgr, /*max_concurrent=*/2);
    auto id = sched.submit(make_request(4, 1, 200));
    auto step = sched.prepare_step();
    EXPECT_EQ(step.n_active(), 1);
    const int free_before = mgr.num_free_blocks();
    EXPECT_LT(free_before, mgr.num_total_blocks());
    sched.commit_step({true}, {0});  // completed
    EXPECT_EQ(mgr.num_free_blocks(), mgr.num_total_blocks());
    EXPECT_EQ(sched.active_size(), 0u);
    EXPECT_EQ(sched.generated(id), 1);
}

TEST(Scheduler, AutoCompleteOnMaxNewTokens) {
    lh::CacheManager mgr(layout(/*blocks=*/8));
    lh::Scheduler sched(mgr, /*max_concurrent=*/1);
    auto id = sched.submit(make_request(4, /*max_new=*/3, 300));
    int steps = 0;
    while (sched.has_work()) {
        auto step = sched.prepare_step();
        if (step.n_active() == 0) break;
        sched.commit_step({false}, {7});
        ++steps;
    }
    EXPECT_EQ(sched.generated(id), 3);
    EXPECT_EQ(steps, 3);
    EXPECT_EQ(mgr.num_free_blocks(), mgr.num_total_blocks());
}

TEST(Scheduler, MixedWorkloadHitsHighOccupancy) {
    // Acceptance gate: a synthetic mixed-length workload should sustain
    // >= 80% occupancy under continuous batching. We submit a stream of
    // requests well in excess of `max_concurrent`, decode each to its
    // declared length, and average the per-step occupancy.
    constexpr int max_concurrent = 8;
    constexpr int n_requests = 50;
    lh::CacheManager mgr(layout(/*blocks=*/64, /*block_size=*/4));
    lh::Scheduler sched(mgr, max_concurrent);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prompt_dist(1, 6);
    std::uniform_int_distribution<int> new_dist(2, 6);
    for (int i = 0; i < n_requests; ++i) {
        sched.submit(make_request(prompt_dist(rng), new_dist(rng),
                                   static_cast<uint32_t>(900 + i)));
    }

    double occ_sum = 0.0;
    int occ_steps = 0;
    while (sched.has_work()) {
        auto step = sched.prepare_step();
        if (step.n_active() == 0) break;
        std::vector<bool> done(static_cast<size_t>(step.n_active()), false);
        std::vector<int32_t> toks(static_cast<size_t>(step.n_active()), 5);
        sched.commit_step(done, toks);
        occ_sum += sched.last_step_occupancy();
        ++occ_steps;
    }
    const double mean_occ = occ_sum / occ_steps;
    // Tail flushes the final few requests through under-filled batches;
    // the bulk of the run should sit very close to 1.0.
    EXPECT_GT(mean_occ, 0.80) << "mean occupancy = " << mean_occ
                              << " across " << occ_steps << " steps";
    EXPECT_EQ(sched.active_size(), 0u);
    EXPECT_EQ(mgr.num_free_blocks(), mgr.num_total_blocks());
}
