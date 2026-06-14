#include "kernels/ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lh {

namespace {

inline int64_t xy_offset(const SelectiveScanConfig& cfg, int b, int t, int d) {
    return ((static_cast<int64_t>(b) * cfg.seq + t) * cfg.d_inner) + d;
}
inline int64_t bc_offset(const SelectiveScanConfig& cfg, int b, int t, int n) {
    return ((static_cast<int64_t>(b) * cfg.seq + t) * cfg.d_state) + n;
}
inline int64_t a_offset(const SelectiveScanConfig& cfg, int d, int n) {
    return static_cast<int64_t>(d) * cfg.d_state + n;
}

}  // namespace

void selective_scan_ref(const float* x, const float* delta, const float* A,
                        const float* B, const float* C, const float* D,
                        float* y, const SelectiveScanConfig& cfg) {
    const int Bsz = cfg.batch;
    const int L = cfg.seq;
    const int Di = cfg.d_inner;
    const int Ds = cfg.d_state;

    // h: [Bsz, Di, Ds] recurrent state, zero-initialized.
    std::vector<float> h(static_cast<size_t>(Bsz) * Di * Ds, 0.0f);

    for (int b = 0; b < Bsz; ++b) {
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < Di; ++d) {
                const float xt = x[xy_offset(cfg, b, t, d)];
                const float dt = delta[xy_offset(cfg, b, t, d)];
                float* hd = h.data() +
                            ((static_cast<int64_t>(b) * Di + d) * Ds);
                float yt = 0.0f;
                for (int n = 0; n < Ds; ++n) {
                    const float a_bar = std::exp(dt * A[a_offset(cfg, d, n)]);
                    const float b_bar = dt * B[bc_offset(cfg, b, t, n)];
                    hd[n] = a_bar * hd[n] + b_bar * xt;
                    yt += C[bc_offset(cfg, b, t, n)] * hd[n];
                }
                if (D) yt += D[d] * xt;
                y[xy_offset(cfg, b, t, d)] = yt;
            }
        }
    }
}

void selective_scan_chunked(const float* x, const float* delta, const float* A,
                            const float* B, const float* C, const float* D,
                            float* y, int chunk_size,
                            const SelectiveScanConfig& cfg) {
    if (chunk_size < 1) chunk_size = 1;
    const int Bsz = cfg.batch;
    const int L = cfg.seq;
    const int Di = cfg.d_inner;
    const int Ds = cfg.d_state;

    std::vector<float> h(static_cast<size_t>(Bsz) * Di * Ds, 0.0f);

    // Within a chunk: hold per-channel discretization tables A_bar, B_bar
    // (sized chunk_size × d_state each) so the within-chunk recurrence can
    // be expressed as a sequence of fp32 reductions against the carry
    // state plus a small lower-triangular matmul. On silicon the LT
    // matmul block becomes a tensor-tile; here we walk it sequentially
    // because correctness is the goal.
    std::vector<float> Abar(static_cast<size_t>(chunk_size) * Ds);
    std::vector<float> Bbar(static_cast<size_t>(chunk_size) * Ds);
    std::vector<float> hh(static_cast<size_t>(Ds));

    for (int b = 0; b < Bsz; ++b) {
        for (int d = 0; d < Di; ++d) {
            // Pull this channel's persistent state into hh.
            float* hd = h.data() + ((static_cast<int64_t>(b) * Di + d) * Ds);
            std::memcpy(hh.data(), hd, Ds * sizeof(float));

            for (int t0 = 0; t0 < L; t0 += chunk_size) {
                const int len = std::min(chunk_size, L - t0);
                // Build per-step Abar / Bbar within the chunk.
                for (int s = 0; s < len; ++s) {
                    const float dt =
                        delta[xy_offset(cfg, b, t0 + s, d)];
                    for (int n = 0; n < Ds; ++n) {
                        Abar[static_cast<int64_t>(s) * Ds + n] =
                            std::exp(dt * A[a_offset(cfg, d, n)]);
                        Bbar[static_cast<int64_t>(s) * Ds + n] =
                            dt * B[bc_offset(cfg, b, t0 + s, n)];
                    }
                }
                // Within-chunk recurrence + readout. Walk steps inside
                // the chunk; the per-step work is per-state-dim
                // reductions identical to GEMV — the SSD silicon path
                // batches these across (batch, channel, chunk).
                for (int s = 0; s < len; ++s) {
                    const float xt = x[xy_offset(cfg, b, t0 + s, d)];
                    float yt = 0.0f;
                    for (int n = 0; n < Ds; ++n) {
                        hh[n] = Abar[static_cast<int64_t>(s) * Ds + n] * hh[n] +
                                Bbar[static_cast<int64_t>(s) * Ds + n] * xt;
                        yt += C[bc_offset(cfg, b, t0 + s, n)] * hh[n];
                    }
                    if (D) yt += D[d] * xt;
                    y[xy_offset(cfg, b, t0 + s, d)] = yt;
                }
            }
            // Persist the carry for the next batch invocation if any.
            std::memcpy(hd, hh.data(), Ds * sizeof(float));
        }
    }
}

}  // namespace lh
