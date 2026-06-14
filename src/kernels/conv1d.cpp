#include "kernels/conv1d.hpp"

#include <cstdint>

namespace lh {

void conv1d_causal_depthwise(const float* x, const float* weight,
                             const float* bias, float* y,
                             const Conv1dConfig& cfg) {
    const int Bsz = cfg.batch;
    const int L = cfg.seq;
    const int C = cfg.channels;
    const int K = cfg.kernel_size;

    for (int b = 0; b < Bsz; ++b) {
        for (int t = 0; t < L; ++t) {
            float* yt = y + ((static_cast<int64_t>(b) * L + t) * C);
            for (int c = 0; c < C; ++c) {
                const float* w = weight + static_cast<int64_t>(c) * K;
                float acc = bias ? bias[c] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    const int s = t - k;
                    if (s < 0) break;  // causal pad with zeros
                    acc += x[((static_cast<int64_t>(b) * L + s) * C) + c] *
                           w[k];
                }
                yt[c] = acc;
            }
        }
    }
}

}  // namespace lh
