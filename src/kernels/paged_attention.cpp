#include "kernels/paged_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lh {

namespace {

inline float resolve_scale(const PagedAttnConfig& cfg) {
    if (cfg.scale > 0.0f) return cfg.scale;
    return 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
}

inline bool mask_allows(const PagedAttnConfig& cfg, int i, int j,
                        int pos_shift) {
    if (cfg.causal && j > pos_shift + i) return false;
    if (cfg.window > 0) {
        const int lo = pos_shift + i - cfg.window + 1;
        if (j < lo) return false;
    }
    if (cfg.chunk > 0) {
        const int q_pos = pos_shift + i;
        if ((q_pos / cfg.chunk) != (j / cfg.chunk)) return false;
    }
    return true;
}

// Per-(request, head) attention over a paged K/V pool. Reads K/V slot-by-
// slot through the block table, which is the cleanest reference: it walks
// each cache position with a single indirection through `block_table` and
// then computes attention identically to SDPA.
void attend_one(const float* Q, const float* K_pool, const float* V_pool,
                const int32_t* block_table, int seq_len,
                int qh, int kvh, int seq_q,
                const PagedAttnConfig& cfg, const PagedCacheLayout& L,
                float* O,
                std::vector<float>& scratch_scores) {
    const float scale = resolve_scale(cfg);
    const int d = cfg.head_dim;
    const int B = L.block_size;
    const int pos_shift = seq_len - seq_q;

    auto kv_ptr = [&](const float* pool, int abs_pos) {
        const int log_block = abs_pos / B;
        const int slot = abs_pos % B;
        const int phys = block_table[log_block];
        return pool + paged_offset(L, phys, kvh, slot, 0);
    };

    if (static_cast<int>(scratch_scores.size()) < seq_len) {
        scratch_scores.resize(static_cast<size_t>(seq_len));
    }

    for (int i = 0; i < seq_q; ++i) {
        const float* q = Q + (static_cast<int64_t>(qh) * seq_q + i) * d;
        const int last = cfg.causal ? std::min(seq_len - 1, pos_shift + i)
                                     : seq_len - 1;
        const int first =
            (cfg.window > 0)
                ? std::max(0, pos_shift + i - cfg.window + 1)
                : 0;

        float* o = O + (static_cast<int64_t>(qh) * seq_q + i) * d;

        if (last < first) {
            for (int t = 0; t < d; ++t) o[t] = 0.0f;
            continue;
        }

        float m = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (int j = first; j <= last; ++j) {
            if (!mask_allows(cfg, i, j, pos_shift)) continue;
            const float* k = kv_ptr(K_pool, j);
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += q[t] * k[t];
            dot *= scale;
            scratch_scores[static_cast<size_t>(j)] = dot;
            if (dot > m) m = dot;
            any = true;
        }
        if (!any) {
            for (int t = 0; t < d; ++t) o[t] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int j = first; j <= last; ++j) {
            if (!mask_allows(cfg, i, j, pos_shift)) continue;
            scratch_scores[static_cast<size_t>(j)] =
                std::exp(scratch_scores[static_cast<size_t>(j)] - m);
            sum += scratch_scores[static_cast<size_t>(j)];
        }
        const float inv = 1.0f / sum;

        for (int t = 0; t < d; ++t) o[t] = 0.0f;
        for (int j = first; j <= last; ++j) {
            if (!mask_allows(cfg, i, j, pos_shift)) continue;
            const float p = scratch_scores[static_cast<size_t>(j)] * inv;
            const float* v = kv_ptr(V_pool, j);
            for (int t = 0; t < d; ++t) o[t] += p * v[t];
        }
    }
}

}  // namespace

void paged_attention(const float* Q, const float* K_pool, const float* V_pool,
                     const int32_t* block_table, int seq_len,
                     float* O, const PagedAttnConfig& cfg,
                     const PagedCacheLayout& L) {
    const int group = cfg.n_q_heads / cfg.n_kv_heads;
    std::vector<float> scratch;
    for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
        const int kvh = qh / group;
        attend_one(Q, K_pool, V_pool, block_table, seq_len, qh, kvh,
                   cfg.seq_q, cfg, L, O, scratch);
    }
}

void paged_attention_batched(const float* Q, const float* K_pool,
                             const float* V_pool,
                             const int32_t* block_tables,
                             int max_blocks_per_req,
                             const int32_t* seq_lens, int n_requests,
                             float* O, const PagedAttnConfig& cfg,
                             const PagedCacheLayout& L) {
    const int group = cfg.n_q_heads / cfg.n_kv_heads;
    const int64_t per_req_q =
        static_cast<int64_t>(cfg.n_q_heads) * cfg.seq_q * cfg.head_dim;

    std::vector<float> scratch;
    for (int r = 0; r < n_requests; ++r) {
        const float* Qr = Q + r * per_req_q;
        float* Or = O + r * per_req_q;
        const int32_t* table = block_tables + r * max_blocks_per_req;
        const int sl = seq_lens[r];
        for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
            const int kvh = qh / group;
            attend_one(Qr, K_pool, V_pool, table, sl, qh, kvh, cfg.seq_q,
                       cfg, L, Or, scratch);
        }
    }
}

}  // namespace lh
