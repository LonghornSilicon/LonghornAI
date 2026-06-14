// LonghornAI — 1D depthwise causal convolution.
//
// The short-filter convolution Mamba applies to each channel before the
// selective scan. "Depthwise" = one filter per channel, no cross-channel
// mixing; "causal" = output at t depends only on inputs at t' <= t,
// implemented by left-padding with zeros.
//
// Layout: x and y are [batch, seq, channels]; weight is [channels, K]
// (one filter of length K per channel). Optional bias is [channels].
//
// On silicon, this is small: per channel an FIR filter of length K (~4)
// per token. It maps onto the vector unit alongside the SSM scan and
// shouldn't need a dedicated engine.
#ifndef LONGHORNAI_KERNELS_CONV1D_HPP
#define LONGHORNAI_KERNELS_CONV1D_HPP

#include <cstdint>

namespace lh {

struct Conv1dConfig {
    int batch = 1;
    int seq = 1;
    int channels = 1;
    int kernel_size = 1;  // K
};

// y[b, t, c] = sum_{k=0..K-1} x[b, t-k, c] * weight[c, k]   (+ bias[c])
// where x[b, t-k, c] = 0 for t-k < 0.
void conv1d_causal_depthwise(const float* x, const float* weight,
                             const float* bias /* may be null */,
                             float* y, const Conv1dConfig& cfg);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_CONV1D_HPP
