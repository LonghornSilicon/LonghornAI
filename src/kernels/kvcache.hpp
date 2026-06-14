// LonghornAI — paged KV cache primitives.
//
// Physical layout of the cache pool, per-request block tables, and the
// append routine that writes new K/V tokens into the right physical block
// slot. The paged-attention kernel (`kernels/paged_attention.hpp`) is the
// other half; this header defines the storage layout it walks.
//
// Pool layout (row-major):
//   K_pool / V_pool: [num_blocks, n_kv_heads, block_size, head_dim]
//
// Per-request block table:
//   block_table[i] = physical block id holding logical positions
//                    [i*block_size, (i+1)*block_size).
//
// All requests in a batch share the same `(num_blocks, block_size,
// n_kv_heads, head_dim)`. Block size is typically 16 (matches vLLM's
// default and is small enough for fine-grained reuse without too much
// per-block bookkeeping overhead).
#ifndef LONGHORNAI_KERNELS_KVCACHE_HPP
#define LONGHORNAI_KERNELS_KVCACHE_HPP

#include <cstdint>

namespace lh {

struct PagedCacheLayout {
    int num_blocks = 0;     // physical capacity of the pool
    int block_size = 16;    // tokens per block
    int n_kv_heads = 1;
    int head_dim = 1;
};

// Linear offset of element (block, head, slot, dim_idx) inside K_pool/V_pool.
inline int64_t paged_offset(const PagedCacheLayout& L, int block, int head,
                            int slot, int d) {
    return ((((static_cast<int64_t>(block) * L.n_kv_heads + head) *
              L.block_size) +
             slot) *
            L.head_dim) +
           d;
}

// Append `seq_new` timesteps of K/V for a single request. The caller has
// already extended the request's `block_table` to cover positions
// [past_len, past_len + seq_new) and zeroed any newly-allocated blocks
// (zero-init keeps the unused slots at the end of the last block from
// polluting attention scores if anyone forgets to mask them).
//   k_new / v_new: [n_kv_heads, seq_new, head_dim]
//   block_table:   length >= ceil((past_len + seq_new) / block_size)
void paged_kv_append(float* k_pool, float* v_pool,
                     const float* k_new, const float* v_new,
                     const int32_t* block_table,
                     const PagedCacheLayout& L,
                     int past_len, int seq_new);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_KVCACHE_HPP
