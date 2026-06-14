// LonghornAI — positional bias and frequency-scaling helpers.
//
// Companion to `rope.hpp`. RoPE itself supports linear (`freq_scale`) and
// NTK-style (`theta_base`) scaling directly; this header provides
// derivation helpers and the additive-bias schemes (ALiBi) that don't fit
// into the RoPE rotation form.
#ifndef LONGHORNAI_KERNELS_POSITIONAL_HPP
#define LONGHORNAI_KERNELS_POSITIONAL_HPP

namespace lh {

// NTK-aware RoPE base. Given an original training context `orig_max_pos`
// and a desired extended context `ext_max_pos`, return the new theta base
// to pass into `rope`. This is the closed-form NTK scaling: scale theta so
// the highest-frequency band rotates by the same amount at the extended
// length as at the original length.
//
//   alpha = ext / orig
//   new_theta = theta_base * alpha^(d / (d - 2))
//
// where d = head_dim. Pass the result as `theta_base` to `rope`.
float ntk_scaled_theta(float theta_base, int orig_max_pos, int ext_max_pos,
                       int head_dim);

// YaRN frequency-band scaling. Computes per-pair inverse frequencies that
// blend "linear" (low-frequency, scale by alpha) and "extrapolation"
// (high-frequency, leave unscaled) regimes, with a ramp in between.
// Writes `head_dim/2` values into `inv_freq_out`. Caller multiplies these
// by position to get the rotation angle, identically to RoPE's `pair_freq`.
//
//   beta_fast / beta_slow define the band edges (typical: 32 / 1).
//   `scale` = ext_max_pos / orig_max_pos.
void yarn_inv_freq(float* inv_freq_out, int head_dim,
                   float theta_base, float scale,
                   float beta_fast = 32.0f, float beta_slow = 1.0f,
                   int orig_max_pos = 2048);

// ALiBi per-head slopes. For `n_heads` heads, fills `slopes_out` with the
// canonical geometric series 2^(-8/n) * 2^(-i*8/n). Used to materialise the
// additive bias `-slope[h] * |q_pos - k_pos|` consumed by attention.
void alibi_slopes(float* slopes_out, int n_heads);

// Materialise an ALiBi additive bias of shape [n_heads, seq_q, seq_k]:
//   bias[h, i, j] = -slope[h] * max(0, q_pos - k_pos)
// where q_pos = i + (seq_k - seq_q) and k_pos = j. The clamp matches the
// causal-decoder convention; for non-causal use, omit the max(0, .).
void alibi_bias(float* bias_out, const float* slopes,
                int n_heads, int seq_q, int seq_k, bool causal = true);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_POSITIONAL_HPP
