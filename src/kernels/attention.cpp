#include "kernels/attention.hpp"

#include <algorithm>
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

// Returns true iff query `i` is allowed to attend to key `j` under the
// optional causal/window/chunk constraints. `pos_shift` is `seq_k - seq_q`.
inline bool mask_allows(const AttnConfig& cfg, int i, int j, int pos_shift) {
    if (cfg.causal && j > pos_shift + i) return false;
    if (cfg.window > 0) {
        const int lo = pos_shift + i - cfg.window + 1;
        if (j < lo) return false;
    }
    if (cfg.chunk > 0) {
        // Chunked attention: queries can only see keys in the same
        // chunk-aligned bucket.
        const int q_pos = pos_shift + i;
        if ((q_pos / cfg.chunk) != (j / cfg.chunk)) return false;
    }
    return true;
}

inline float bias_at(const AttnConfig& cfg, int qh, int i, int j) {
    if (!cfg.bias) return 0.0f;
    if (cfg.bias_per_head) {
        return cfg
            .bias[((static_cast<int64_t>(qh) * cfg.seq_q + i) * cfg.seq_k) + j];
    }
    return cfg.bias[(static_cast<int64_t>(i) * cfg.seq_k) + j];
}

// Last attended key index for query `i` under causal + window. `chunk` is
// applied per-key inside the loop because it isn't a simple upper bound.
inline int last_key(const AttnConfig& cfg, int i, int pos_shift) {
    int last = cfg.seq_k - 1;
    if (cfg.causal) last = std::min(last, pos_shift + i);
    return last;
}

inline int first_key(const AttnConfig& cfg, int i, int pos_shift) {
    int first = 0;
    if (cfg.window > 0) {
        const int lo = pos_shift + i - cfg.window + 1;
        if (lo > 0) first = lo;
    }
    return first;
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
                const int last = last_key(cfg, i, pos_shift);
                const int first = first_key(cfg, i, pos_shift);

                float m = -std::numeric_limits<float>::infinity();
                bool any = false;
                for (int j = first; j <= last; ++j) {
                    if (!mask_allows(cfg, i, j, pos_shift)) continue;
                    const float* k = K + row_offset(b, kvh, j, cfg.n_kv_heads,
                                                    cfg.seq_k, d);
                    float dot = 0.0f;
                    for (int t = 0; t < d; ++t) dot += q[t] * k[t];
                    dot = dot * scale + bias_at(cfg, qh, i, j);
                    scores[static_cast<size_t>(j)] = dot;
                    if (dot > m) m = dot;
                    any = true;
                }

                float* o = O + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
                if (!any) {
                    // No allowed keys (e.g. windowed beyond start): emit 0.
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
                    const float* v = V + row_offset(b, kvh, j, cfg.n_kv_heads,
                                                    cfg.seq_k, d);
                    for (int t = 0; t < d; ++t) o[t] += p * v[t];
                }
            }
        }
    }
}

namespace {

// One-pass online attention over a *contiguous* K range [k_lo, k_hi],
// honouring the cfg's mask constraints. Writes (m, l) and the unnormalised
// accumulator `acc` so callers can compose splits via merge_partials().
struct OnlineState {
    float m;     // running max
    float l;     // running normaliser
};

inline OnlineState online_block(const AttnConfig& cfg, const float* Q,
                                const float* K, const float* V,
                                int b, int qh, int kvh, int i, int pos_shift,
                                int k_lo, int k_hi, int block_k,
                                float* acc /* len d */, std::vector<float>& blk) {
    const float scale = resolve_scale(cfg);
    const int d = cfg.head_dim;
    OnlineState s{-std::numeric_limits<float>::infinity(), 0.0f};
    for (int t = 0; t < d; ++t) acc[t] = 0.0f;

    const float* q = Q + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
    if (block_k < 1) block_k = 1;

    for (int j0 = k_lo; j0 <= k_hi; j0 += block_k) {
        const int jmax = std::min(j0 + block_k - 1, k_hi);
        float block_max = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (int j = j0; j <= jmax; ++j) {
            if (!mask_allows(cfg, i, j, pos_shift)) {
                blk[static_cast<size_t>(j - j0)] =
                    -std::numeric_limits<float>::infinity();
                continue;
            }
            const float* k = K + row_offset(b, kvh, j, cfg.n_kv_heads,
                                            cfg.seq_k, d);
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += q[t] * k[t];
            dot = dot * scale + bias_at(cfg, qh, i, j);
            blk[static_cast<size_t>(j - j0)] = dot;
            if (dot > block_max) block_max = dot;
            any = true;
        }
        if (!any) continue;

        const float m_new = (block_max > s.m) ? block_max : s.m;
        const float correction =
            (s.m == -std::numeric_limits<float>::infinity())
                ? 0.0f
                : std::exp(s.m - m_new);
        s.l *= correction;
        for (int t = 0; t < d; ++t) acc[t] *= correction;

        for (int j = j0; j <= jmax; ++j) {
            const float raw = blk[static_cast<size_t>(j - j0)];
            if (raw == -std::numeric_limits<float>::infinity()) continue;
            const float p = std::exp(raw - m_new);
            s.l += p;
            const float* v = V + row_offset(b, kvh, j, cfg.n_kv_heads,
                                            cfg.seq_k, d);
            for (int t = 0; t < d; ++t) acc[t] += p * v[t];
        }
        s.m = m_new;
    }
    return s;
}

}  // namespace

void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg, int block_k) {
    const int d = cfg.head_dim;
    const int pos_shift = cfg.seq_k - cfg.seq_q;
    const int group = cfg.n_q_heads / cfg.n_kv_heads;

    std::vector<float> acc(static_cast<size_t>(d));
    std::vector<float> blk(static_cast<size_t>(std::max(1, block_k)));

    for (int b = 0; b < cfg.batch; ++b) {
        for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
            const int kvh = qh / group;
            for (int i = 0; i < cfg.seq_q; ++i) {
                const int k_lo = first_key(cfg, i, pos_shift);
                const int k_hi = last_key(cfg, i, pos_shift);
                float* o = O + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
                if (k_hi < k_lo) {  // window exiles all keys for this query
                    for (int t = 0; t < d; ++t) o[t] = 0.0f;
                    continue;
                }
                auto s = online_block(cfg, Q, K, V, b, qh, kvh, i, pos_shift,
                                      k_lo, k_hi, block_k, acc.data(), blk);
                const float inv = (s.l > 0.0f) ? 1.0f / s.l : 0.0f;
                for (int t = 0; t < d; ++t) o[t] = acc[t] * inv;
            }
        }
    }
}

void flash_decode_attention(const float* Q, const float* K, const float* V,
                            float* O, const AttnConfig& cfg, int block_k,
                            int splits) {
    if (splits < 1) splits = 1;
    const int d = cfg.head_dim;
    const int pos_shift = cfg.seq_k - cfg.seq_q;
    const int group = cfg.n_q_heads / cfg.n_kv_heads;

    std::vector<float> blk(static_cast<size_t>(std::max(1, block_k)));
    std::vector<float> part_acc(static_cast<size_t>(d));
    std::vector<float> merged_acc(static_cast<size_t>(d));

    for (int b = 0; b < cfg.batch; ++b) {
        for (int qh = 0; qh < cfg.n_q_heads; ++qh) {
            const int kvh = qh / group;
            for (int i = 0; i < cfg.seq_q; ++i) {
                const int k_lo = first_key(cfg, i, pos_shift);
                const int k_hi = last_key(cfg, i, pos_shift);
                float* o = O + row_offset(b, qh, i, cfg.n_q_heads, cfg.seq_q, d);
                if (k_hi < k_lo) {
                    for (int t = 0; t < d; ++t) o[t] = 0.0f;
                    continue;
                }
                const int total = k_hi - k_lo + 1;
                const int per = (total + splits - 1) / splits;

                // Merge partial (m, l, acc) states from each split into
                // (m_g, l_g, acc_g) using the standard online-softmax rule.
                float m_g = -std::numeric_limits<float>::infinity();
                float l_g = 0.0f;
                for (int t = 0; t < d; ++t) merged_acc[t] = 0.0f;

                for (int s = 0; s < splits; ++s) {
                    const int s_lo = k_lo + s * per;
                    if (s_lo > k_hi) break;
                    const int s_hi = std::min(s_lo + per - 1, k_hi);
                    auto st = online_block(cfg, Q, K, V, b, qh, kvh, i,
                                           pos_shift, s_lo, s_hi, block_k,
                                           part_acc.data(), blk);
                    if (st.l <= 0.0f) continue;  // empty after masking
                    const float m_new = std::max(m_g, st.m);
                    const float c_g =
                        (m_g == -std::numeric_limits<float>::infinity())
                            ? 0.0f
                            : std::exp(m_g - m_new);
                    const float c_s = std::exp(st.m - m_new);
                    for (int t = 0; t < d; ++t) {
                        merged_acc[t] = merged_acc[t] * c_g +
                                        part_acc[t] * c_s;
                    }
                    l_g = l_g * c_g + st.l * c_s;
                    m_g = m_new;
                }

                const float inv = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
                for (int t = 0; t < d; ++t) o[t] = merged_acc[t] * inv;
            }
        }
    }
}

void cross_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg) {
    // Cross-attention is acausal by definition; reuse SDPA but defensively
    // override `causal` so callers who copy a config from a self-attention
    // path don't accidentally apply a causal mask across streams.
    AttnConfig c = cfg;
    c.causal = false;
    sdpa(Q, K, V, O, c);
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
