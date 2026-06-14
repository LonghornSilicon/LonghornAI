// LonghornAI — selective state-space model kernels (Mamba / Mamba-2).
//
// The selective scan is the central operator behind Mamba. Per (batch,
// channel) it integrates an input-dependent linear ODE:
//
//   discretize:  A_bar = exp(delta * A)        (per channel: [d_state])
//                B_bar = delta * B             (input-dependent)
//   recurrence:  h[t]  = A_bar[t] * h[t-1] + B_bar[t] * x[t]
//   readout:     y[t]  = C[t] @ h[t]   (+ D * x[t]  optional skip)
//
// `selective_scan_ref` is the dead-simple sequential reference: walk
// timesteps, update h ∈ R^{d_state} per channel, dot with C and add the
// skip. This is the Mamba S6 numerics specification.
//
// `selective_scan_chunked` is the Mamba-2 / SSD recipe: split the sequence
// into chunks of `chunk_size`; within a chunk express the contribution as
// a small lower-triangular semi-separable matmul; across chunks pass the
// recurrent state forward. This is the algorithmic shape that maps onto
// the same tensor unit / GEMM datapath as attention — the silicon-side
// motivation for picking SSD over the naive scan.
#ifndef LONGHORNAI_KERNELS_SSM_HPP
#define LONGHORNAI_KERNELS_SSM_HPP

#include <cstdint>

namespace lh {

struct SelectiveScanConfig {
    int batch = 1;
    int seq = 1;
    int d_inner = 1;   // channel count
    int d_state = 1;   // state dim per channel
};

// Sequential reference scan. Inputs:
//   x:     fp32 [batch, seq, d_inner]
//   delta: fp32 [batch, seq, d_inner]    discretization step (>= 0)
//   A:     fp32 [d_inner, d_state]       per-channel state matrix (continuous-time
//                                        eigenvalues; in Mamba these are < 0)
//   B:     fp32 [batch, seq, d_state]    input-dependent
//   C:     fp32 [batch, seq, d_state]    input-dependent
//   D:     fp32 [d_inner] or nullptr     skip-connection coefficients
// Output:
//   y:     fp32 [batch, seq, d_inner]
void selective_scan_ref(const float* x, const float* delta, const float* A,
                        const float* B, const float* C, const float* D,
                        float* y, const SelectiveScanConfig& cfg);

// Chunked SSD scan. Same numerics as `selective_scan_ref` (must agree
// within fp tolerance for any chunk_size >= 1). The structure is what
// matters for silicon: each chunk is a sequence of small fp32 reductions
// that map onto the existing tensor unit.
void selective_scan_chunked(const float* x, const float* delta,
                            const float* A, const float* B, const float* C,
                            const float* D, float* y, int chunk_size,
                            const SelectiveScanConfig& cfg);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_SSM_HPP
