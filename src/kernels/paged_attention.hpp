// LonghornAI — paged attention.
//
// Variant of SDPA that reads K/V from a paged pool indexed by per-request
// block tables instead of a contiguous `[seq_k, head_dim]` tensor.
// Functionally equivalent to running SDPA over a contiguous gather of the
// pool blocks; the value of paging is that the pool can be shared across
// requests with different sequence lengths and across requests that share
// a prefix.
//
// Layout: K/V pool follows `PagedCacheLayout`. For each request we have:
//   - Q:           [n_q_heads, seq_q, head_dim]
//   - block_table: [num_logical_blocks]   (logical -> physical block id)
//   - seq_len:     int                    (valid positions in the cache)
//
// The kernel honours the same optional bias / window / chunk constraints as
// `AttnConfig`. Causal masking is handled relative to seq_len: the last Q
// position aligns with cache position seq_len-1.
#ifndef LONGHORNAI_KERNELS_PAGED_ATTENTION_HPP
#define LONGHORNAI_KERNELS_PAGED_ATTENTION_HPP

#include <cstdint>

#include "kernels/attention.hpp"
#include "kernels/kvcache.hpp"

namespace lh {

struct PagedAttnConfig {
    int n_q_heads = 1;
    int n_kv_heads = 1;
    int seq_q = 1;
    int head_dim = 1;
    bool causal = true;
    float scale = 0.0f;  // 1/sqrt(head_dim) when <= 0
    int window = 0;
    int chunk = 0;
};

// Single-request paged attention. `seq_len` is the number of valid cache
// positions; the kernel attends Q over keys [0, seq_len). Causal masking
// aligns the last query with cache position `seq_len - 1`.
void paged_attention(const float* Q,
                     const float* K_pool, const float* V_pool,
                     const int32_t* block_table,
                     int seq_len,
                     float* O,
                     const PagedAttnConfig& cfg,
                     const PagedCacheLayout& L);

// Batched paged attention. `n_requests` requests share `K_pool`/`V_pool`.
// Each request has its own `block_table` (stored in row `r` of
// `block_tables`, stride `max_blocks_per_req`) and its own `seq_lens[r]`.
// Q is laid out [n_requests, n_q_heads, seq_q, head_dim] (uniform seq_q —
// suitable for prefill batches and decode steps).
void paged_attention_batched(const float* Q,
                             const float* K_pool, const float* V_pool,
                             const int32_t* block_tables,
                             int max_blocks_per_req,
                             const int32_t* seq_lens,
                             int n_requests,
                             float* O,
                             const PagedAttnConfig& cfg,
                             const PagedCacheLayout& L);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_PAGED_ATTENTION_HPP
