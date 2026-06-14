#include "kernels/rwkv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lh {

namespace {

inline int64_t bcs_offset(const WkvConfig& cfg, int b, int t, int c) {
    return ((static_cast<int64_t>(b) * cfg.seq + t) * cfg.n_channels) + c;
}

}  // namespace

void wkv(const float* k, const float* v, const float* w, const float* u,
         float* y, const WkvConfig& cfg) {
    const int Bsz = cfg.batch;
    const int L = cfg.seq;
    const int C = cfg.n_channels;

    for (int b = 0; b < Bsz; ++b) {
        for (int c = 0; c < C; ++c) {
            // Running state per (batch, channel). Initialize p to -inf so
            // the very first step can subtract a clean max.
            float a = 0.0f;   // weighted-sum-of-v in exp-shifted space
            float b_ = 0.0f;  // weighted-sum-of-weights
            float p = -std::numeric_limits<float>::infinity();
            const float wc = w[c];
            const float uc = u[c];

            for (int t = 0; t < L; ++t) {
                const float kt = k[bcs_offset(cfg, b, t, c)];
                const float vt = v[bcs_offset(cfg, b, t, c)];

                // Output at step t. The stored state (a, b, p) represents
                // the running sums N_{t-1}, L_{t-1} (no bonus, no extra
                // decay) with log-shift p. To weight that against the
                // bonus on step t we first apply one step of decay (shift
                // becomes p - w), then take a stable max against the
                // bonus's natural exponent (u + k_t).
                const float p_decayed = p - wc;
                const float ukt = uc + kt;
                const float q = std::max(p_decayed, ukt);
                const float ep = std::exp(p_decayed - q);
                const float ek = std::exp(ukt - q);
                const float num = ep * a + ek * vt;
                const float den = ep * b_ + ek;
                y[bcs_offset(cfg, b, t, c)] =
                    (den > 0.0f) ? (num / den) : 0.0f;

                // Update (a, b, p) to represent N_t, L_t (state with
                // step t's *non-bonus* contribution included). The
                // already-decayed state (shift p - w) combines with the
                // new term at log-weight k_t.
                const float p_new = std::max(p_decayed, kt);
                const float ep2 = std::exp(p_decayed - p_new);
                const float ek2 = std::exp(kt - p_new);
                a = ep2 * a + ek2 * vt;
                b_ = ep2 * b_ + ek2;
                p = p_new;
            }
        }
    }
}

}  // namespace lh
