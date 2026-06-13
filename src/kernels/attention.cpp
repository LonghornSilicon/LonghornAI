#include "kernels/attention.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lh {

namespace {
inline float resolve_scale(const AttnConfig& cfg) {
    if (cfg.scale > 0.0f) return cfg.scale;
    return 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
}

// Linear offset of head `h`, sequence `s` within (batch `b`) for a tensor with
// `n_heads` heads and `seq` timesteps of width `head_dim`.
inline int64_t row_offset(int b, int h, int s, int n_heads, int seq,
                          int head_dim) {
    return (((static_cast<int64_t>(b) * n_heads + h) * seq) + s) *
           head_dim;
}
}  // namespace

void sdpa(const float* Q, const float* K, const float* V, float* O,
          const AttnConfig& cfg) {
    const float scale = resolve_scale(cfg);
    const int group = cfg.n_q_heads / cfg.n_kv_heads;
    const int d = cfg.head_dim;
    const int pos_shift = cfg.seq_k - cfg.seq_q;

    std::vector<float> scores(static_cast<size_t>(cfg.seq_k));

    for (int b = 0; b < cfg.batch; ++b) {
        for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
            const int kvh = qh / group;
            for (int i = 0; i < cfg.seq_q; ++i) {
                const float* q = Q + row_offset(b, qh, i, cfg.n_q_heads,
                                                cfg.seq_q, d);
                const int last = cfg.causal ? (pos_shift + i) : (cfg.seq_k - 1);

                float m = -std::numeric_limits<float>::infinity();
                for (int j = 0; j <= last; ++j) {
                    const float* k = K + row_offset(b, kvh, j, cfg.n_kv_heads,
                                                    cfg.seq_k, d);
                    float dot = 0.0f;
                    for (int t = 0; t < d; ++t) dot += q[t] * k[t];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > m) m = dot;
                }

                float sum = 0.0f;
                for (int j = 0; j <= last; ++j) {
                    scores[j] = std::exp(scores[j] - m);
                    sum += scores[j];
                }
                const float inv = 1.0f / sum;

                float* o = O + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
                for (int t = 0; t < d; ++t) o[t] = 0.0f;
                for (int j = 0; j <= last; ++j) {
                    const float p = scores[j] * inv;
                    const float* v = V + row_offset(b, kvh, j, cfg.n_kv_heads,
                                                    cfg.seq_k, d);
                    for (int t = 0; t < d; ++t) o[t] += p * v[t];
                }
            }
        }
    }
}

void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg, int block_k) {
    const float scale = resolve_scale(cfg);
    const int group = cfg.n_q_heads / cfg.n_kv_heads;
    const int d = cfg.head_dim;
    const int pos_shift = cfg.seq_k - cfg.seq_q;
    if (block_k < 1) block_k = 1;

    std::vector<float> acc(static_cast<size_t>(d));
    std::vector<float> blk(static_cast<size_t>(block_k));

    for (int b = 0; b < cfg.batch; ++b) {
        for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
            const int kvh = qh / group;
            for (int i = 0; i < cfg.seq_q; ++i) {
                const float* q = Q + row_offset(b, qh, i, cfg.n_q_heads,
                                                cfg.seq_q, d);
                const int last = cfg.causal ? (pos_shift + i) : (cfg.seq_k - 1);

                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                for (int t = 0; t < d; ++t) acc[t] = 0.0f;

                for (int j0 = 0; j0 <= last; j0 += block_k) {
                    const int jmax = (j0 + block_k - 1 < last) ? j0 + block_k - 1
                                                               : last;
                    // Score this key block and find its local max.
                    float block_max = -std::numeric_limits<float>::infinity();
                    for (int j = j0; j <= jmax; ++j) {
                        const float* k = K + row_offset(b, kvh, j,
                                                        cfg.n_kv_heads,
                                                        cfg.seq_k, d);
                        float dot = 0.0f;
                        for (int t = 0; t < d; ++t) dot += q[t] * k[t];
                        dot *= scale;
                        blk[static_cast<size_t>(j - j0)] = dot;
                        if (dot > block_max) block_max = dot;
                    }

                    const float m_new = (block_max > m) ? block_max : m;
                    const float correction = std::exp(m - m_new);
                    l *= correction;
                    for (int t = 0; t < d; ++t) acc[t] *= correction;

                    for (int j = j0; j <= jmax; ++j) {
                        const float p =
                            std::exp(blk[static_cast<size_t>(j - j0)] - m_new);
                        l += p;
                        const float* v = V + row_offset(b, kvh, j,
                                                        cfg.n_kv_heads,
                                                        cfg.seq_k, d);
                        for (int t = 0; t < d; ++t) acc[t] += p * v[t];
                    }
                    m = m_new;
                }

                float* o = O + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
                const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int t = 0; t < d; ++t) o[t] = acc[t] * inv;
            }
        }
    }
}

void kv_cache_append(float* k_cache, float* v_cache,
                     const float* k_new, const float* v_new,
                     int n_kv_heads, int head_dim, int max_seq,
                     int past_len, int seq_new) {
    for (int h = 0; h < n_kv_heads; ++h) {
        for (int s = 0; s < seq_new; ++s) {
            const int64_t dst =
                ((static_cast<int64_t>(h) * max_seq) + (past_len + s)) * head_dim;
            const int64_t src =
                ((static_cast<int64_t>(h) * seq_new) + s) * head_dim;
            for (int t = 0; t < head_dim; ++t) {
                k_cache[dst + t] = k_new[src + t];
                v_cache[dst + t] = v_new[src + t];
            }
        }
    }
}

}  // namespace lh
