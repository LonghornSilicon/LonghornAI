// LonghornAI — INT8-quantized paged KV cache.
//
// Same paged layout as `kvcache.hpp` but the K/V values are stored as
// signed int8 with one fp32 scale per (block, head, slot) row.
//
// Storage:
//   K_pool / V_pool: int8_t [num_blocks, n_kv_heads, block_size, head_dim]
//   K_scales / V_scales:
//                    float  [num_blocks, n_kv_heads, block_size]
//
// Quantization scheme is per-row symmetric: for each head_dim row,
//   scale = max_abs / 127     (or 1 when max_abs == 0)
//   stored = round(val / scale)
// On read,  val = stored * scale.
//
// This is the simplest scheme that recovers most of the fp32 fidelity at
// half the bandwidth; bandwidth wins compound with KV size, so this is
// the right baseline for MQA/GQA decode where KV reads dominate.
#ifndef LONGHORNAI_KERNELS_KVCACHE_QUANT_HPP
#define LONGHORNAI_KERNELS_KVCACHE_QUANT_HPP

#include <cstdint>

#include "kernels/kvcache.hpp"
#include "kernels/paged_attention.hpp"

namespace lh {

// Linear offset of the per-row scale at (block, head, slot) inside a
// scale tensor of shape [num_blocks, n_kv_heads, block_size].
inline int64_t paged_scale_offset(const PagedCacheLayout& L, int block,
                                  int head, int slot) {
    return ((static_cast<int64_t>(block) * L.n_kv_heads + head) *
            L.block_size) +
           slot;
}

// Append `seq_new` timesteps of K/V into the int8 pool. Quantizes per-row
// symmetric using max(abs(.)) / 127; writes the scale alongside.
//   k_new / v_new: fp32 [n_kv_heads, seq_new, head_dim]
void paged_kv_append_q8(int8_t* k_pool, int8_t* v_pool,
                        float* k_scales, float* v_scales,
                        const float* k_new, const float* v_new,
                        const int32_t* block_table,
                        const PagedCacheLayout& L,
                        int past_len, int seq_new);

// Paged attention reading from an int8-quantized KV pool. Functionally
// identical to `paged_attention` modulo the int8 quantization noise on K
// and V; tolerance is documented per-run.
void paged_attention_q8(const float* Q,
                        const int8_t* K_pool, const int8_t* V_pool,
                        const float* K_scales, const float* V_scales,
                        const int32_t* block_table,
                        int seq_len, float* O,
                        const PagedAttnConfig& cfg,
                        const PagedCacheLayout& L);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_KVCACHE_QUANT_HPP
