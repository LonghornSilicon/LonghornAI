#include "kernels/kvcache_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lh {

namespace {

inline int8_t saturate_round(float v) {
    if (v >= 127.0f) return 127;
    if (v <= -127.0f) return -127;
    // Banker's rounding via lrintf is fine for our purposes; the bias
    // matters less than the per-row scale in practice.
    int r = static_cast<int>(std::lrintf(v));
    if (r > 127) r = 127;
    if (r < -127) r = -127;
    return static_cast<int8_t>(r);
}

inline void quantize_row(const float* src, int8_t* dst, float* scale_out,
                         int d) {
    float max_abs = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float a = std::fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < d; ++i) dst[i] = saturate_round(src[i] * inv);
    *scale_out = scale;
}

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

}  // namespace

void paged_kv_append_q8(int8_t* k_pool, int8_t* v_pool, float* k_scales,
                        float* v_scales, const float* k_new, const float* v_new,
                        const int32_t* block_table, const PagedCacheLayout& L,
                        int past_len, int seq_new) {
    if (seq_new <= 0) return;
    const int B = L.block_size;
    const int H = L.n_kv_heads;
    const int D = L.head_dim;

    for (int s = 0; s < seq_new; ++s) {
        const int abs_pos = past_len + s;
        const int phys = block_table[abs_pos / B];
        const int slot = abs_pos % B;
        for (int h = 0; h < H; ++h) {
            const int64_t dst = paged_offset(L, phys, h, slot, 0);
            const int64_t scale_off = paged_scale_offset(L, phys, h, slot);
            const int64_t src =
                ((static_cast<int64_t>(h) * seq_new) + s) * D;
            quantize_row(k_new + src, k_pool + dst, k_scales + scale_off, D);
            quantize_row(v_new + src, v_pool + dst, v_scales + scale_off, D);
        }
    }
}

void paged_attention_q8(const float* Q, const int8_t* K_pool,
                        const int8_t* V_pool, const float* K_scales,
                        const float* V_scales, const int32_t* block_table,
                        int seq_len, float* O, const PagedAttnConfig& cfg,
                        const PagedCacheLayout& L) {
    const float scale = resolve_scale(cfg);
    const int group = cfg.n_q_heads / cfg.n_kv_heads;
    const int d = cfg.head_dim;
    const int B = L.block_size;
    const int pos_shift = seq_len - cfg.seq_q;

    std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);

    auto k_row = [&](int abs_pos, int kvh, float& out_scale) {
        const int phys = block_table[abs_pos / B];
        const int slot = abs_pos % B;
        out_scale = K_scales[paged_scale_offset(L, phys, kvh, slot)];
        return K_pool + paged_offset(L, phys, kvh, slot, 0);
    };
    auto v_row = [&](int abs_pos, int kvh, float& out_scale) {
        const int phys = block_table[abs_pos / B];
        const int slot = abs_pos % B;
        out_scale = V_scales[paged_scale_offset(L, phys, kvh, slot)];
        return V_pool + paged_offset(L, phys, kvh, slot, 0);
    };

    for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
        const int kvh = qh / group;
        for (int i = 0; i < cfg.seq_q; ++i) {
            const float* q = Q + (static_cast<int64_t>(qh) * cfg.seq_q + i) * d;
            const int last = cfg.causal
                                 ? std::min(seq_len - 1, pos_shift + i)
                                 : seq_len - 1;
            const int first = (cfg.window > 0)
                                  ? std::max(0, pos_shift + i - cfg.window + 1)
                                  : 0;
            float* o = O + (static_cast<int64_t>(qh) * cfg.seq_q + i) * d;
            if (last < first) {
                for (int t = 0; t < d; ++t) o[t] = 0.0f;
                continue;
            }

            float m = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int j = first; j <= last; ++j) {
                if (!mask_allows(cfg, i, j, pos_shift)) continue;
                float ks = 0.0f;
                const int8_t* k = k_row(j, kvh, ks);
                float dot = 0.0f;
                for (int t = 0; t < d; ++t) dot += q[t] * (k[t] * ks);
                dot *= scale;
                scores[static_cast<size_t>(j)] = dot;
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
                scores[static_cast<size_t>(j)] =
                    std::exp(scores[static_cast<size_t>(j)] - m);
                sum += scores[static_cast<size_t>(j)];
            }
            const float inv = 1.0f / sum;

            for (int t = 0; t < d; ++t) o[t] = 0.0f;
            for (int j = first; j <= last; ++j) {
                if (!mask_allows(cfg, i, j, pos_shift)) continue;
                const float p = scores[static_cast<size_t>(j)] * inv;
                float vs = 0.0f;
                const int8_t* v = v_row(j, kvh, vs);
                for (int t = 0; t < d; ++t) o[t] += p * (v[t] * vs);
            }
        }
    }
}

}  // namespace lh
