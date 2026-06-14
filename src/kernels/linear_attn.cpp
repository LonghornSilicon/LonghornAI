#include "kernels/linear_attn.hpp"

#include <cstdint>
#include <vector>

namespace lh {

namespace {

inline int64_t qk_off(const LinearAttnConfig& cfg, int b, int h, int t) {
    return (((static_cast<int64_t>(b) * cfg.n_heads + h) * cfg.seq) + t) *
           cfg.d_feat;
}
inline int64_t v_off(const LinearAttnConfig& cfg, int b, int h, int t) {
    return (((static_cast<int64_t>(b) * cfg.n_heads + h) * cfg.seq) + t) *
           cfg.d_v;
}

}  // namespace

void linear_attention(const float* Q, const float* K, const float* V,
                      float* y, const LinearAttnConfig& cfg) {
    const int Bsz = cfg.batch;
    const int H = cfg.n_heads;
    const int L = cfg.seq;
    const int Df = cfg.d_feat;
    const int Dv = cfg.d_v;

    // Per (batch, head) state buffers, allocated outside the inner loop.
    std::vector<float> S(static_cast<size_t>(Df) * Dv);
    std::vector<float> z(static_cast<size_t>(Df));

    for (int b = 0; b < Bsz; ++b) {
        for (int h = 0; h < H; ++h) {
            std::fill(S.begin(), S.end(), 0.0f);
            std::fill(z.begin(), z.end(), 0.0f);

            for (int t = 0; t < L; ++t) {
                const float* qt = Q + qk_off(cfg, b, h, t);
                const float* kt = K + qk_off(cfg, b, h, t);
                const float* vt = V + v_off(cfg, b, h, t);

                // Rank-1 update: S += k_t ⊗ v_t; z += k_t.
                for (int f = 0; f < Df; ++f) {
                    const float kf = kt[f];
                    z[static_cast<size_t>(f)] += kf;
                    float* row = S.data() + static_cast<int64_t>(f) * Dv;
                    for (int d = 0; d < Dv; ++d) row[d] += kf * vt[d];
                }

                // y_t = q_tᵀ S (/ q_tᵀ z if normalize).
                float* yt = y + v_off(cfg, b, h, t);
                for (int d = 0; d < Dv; ++d) yt[d] = 0.0f;
                float zsum = 0.0f;
                for (int f = 0; f < Df; ++f) {
                    const float qf = qt[f];
                    zsum += qf * z[static_cast<size_t>(f)];
                    const float* row = S.data() + static_cast<int64_t>(f) * Dv;
                    for (int d = 0; d < Dv; ++d) yt[d] += qf * row[d];
                }
                if (cfg.normalize) {
                    const float inv = (zsum > 0.0f) ? (1.0f / zsum) : 0.0f;
                    for (int d = 0; d < Dv; ++d) yt[d] *= inv;
                }
            }
        }
    }
}

}  // namespace lh
