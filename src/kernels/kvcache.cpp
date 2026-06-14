#include "kernels/kvcache.hpp"

#include <cstring>

namespace lh {

void paged_kv_append(float* k_pool, float* v_pool,
                     const float* k_new, const float* v_new,
                     const int32_t* block_table,
                     const PagedCacheLayout& L,
                     int past_len, int seq_new) {
    if (seq_new <= 0) return;
    const int B = L.block_size;
    const int H = L.n_kv_heads;
    const int D = L.head_dim;

    for (int s = 0; s < seq_new; ++s) {
        const int abs_pos = past_len + s;
        const int log_block = abs_pos / B;
        const int slot = abs_pos % B;
        const int phys = block_table[log_block];
        for (int h = 0; h < H; ++h) {
            const int64_t dst = paged_offset(L, phys, h, slot, 0);
            const int64_t src =
                ((static_cast<int64_t>(h) * seq_new) + s) * D;
            std::memcpy(k_pool + dst, k_new + src,
                        static_cast<size_t>(D) * sizeof(float));
            std::memcpy(v_pool + dst, v_new + src,
                        static_cast<size_t>(D) * sizeof(float));
        }
    }
}

}  // namespace lh
