// LonghornAI — RWKV WKV recurrence.
//
// Numerically stable WKV operator from RWKV v5/v6. The mathematical form
// computes a weighted decay-attention over past keys/values with a "bonus"
// weight on the current step:
//
//   wkv_t = (sum_{i<t} exp(-(t-i)*w + k_i) * v_i + exp(u + k_t) * v_t) /
//           (sum_{i<t} exp(-(t-i)*w + k_i) + exp(u + k_t))
//
// The recurrence form maintains running (a, b, p) per channel:
//   p:  log of the running max exponent (for stability)
//   a:  weighted sum of v in exp-shifted space
//   b:  weighted sum of weights in exp-shifted space
//
// The kernel layout is per-channel parallel: each of `n_channels`
// independent "head channels" maintains its own (a, b, p) state. This is
// the silicon-relevant shape — RWKV's state is small per channel and the
// channels run independently, so the kernel maps onto a vector-lane
// engine rather than a tensor unit.
//
// Out of scope:
//   - RWKV v7 (state matrix W; needs a different recurrence shape).
//   - The full RWKV block (time-mix + channel-mix); those wrap the WKV
//     core with linear projections and a gating step that compose as
//     existing linear / activation kernels.
#ifndef LONGHORNAI_KERNELS_RWKV_HPP
#define LONGHORNAI_KERNELS_RWKV_HPP

#include <cstdint>

namespace lh {

struct WkvConfig {
    int batch = 1;
    int seq = 1;
    int n_channels = 1;
};

// Numerically stable WKV per-channel.
//   k:       fp32 [batch, seq, n_channels]   key (log-space scalar per ch)
//   v:       fp32 [batch, seq, n_channels]   value
//   w:       fp32 [n_channels]               time decay (>= 0); larger w =
//                                            faster forgetting
//   u:       fp32 [n_channels]               bonus weight on current step
//   y:       fp32 [batch, seq, n_channels]   (output)
//
// Initial state is zero. The kernel is single-pass left-to-right; the
// recurrence is per-(batch, channel) and trivially parallel across both.
void wkv(const float* k, const float* v, const float* w, const float* u,
         float* y, const WkvConfig& cfg);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_RWKV_HPP
