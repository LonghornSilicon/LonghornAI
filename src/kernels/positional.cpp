#include "kernels/positional.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lh {

float ntk_scaled_theta(float theta_base, int orig_max_pos, int ext_max_pos,
                       int head_dim) {
    if (orig_max_pos <= 0 || ext_max_pos <= orig_max_pos || head_dim <= 2) {
        return theta_base;
    }
    const float alpha = static_cast<float>(ext_max_pos) /
                        static_cast<float>(orig_max_pos);
    const float exp_factor =
        static_cast<float>(head_dim) / static_cast<float>(head_dim - 2);
    return theta_base * std::pow(alpha, exp_factor);
}

namespace {

// Position threshold at which a given inverse frequency completes one full
// rotation: pos_at_2pi = (2*pi) / inv_freq * (orig_max_pos / (2*pi))
//                     = orig_max_pos / (inv_freq * orig_max_pos / (2*pi)).
// In the YaRN paper this is expressed as the number of "wavelengths" inside
// `orig_max_pos`, which is what the ramp reads against.
inline float wavelengths_in_orig(float inv_freq, int orig_max_pos) {
    return static_cast<float>(orig_max_pos) /
           (inv_freq * static_cast<float>(orig_max_pos) / (2.0f * 3.14159265358979323846f));
}

}  // namespace

void yarn_inv_freq(float* inv_freq_out, int head_dim, float theta_base,
                   float scale, float beta_fast, float beta_slow,
                   int orig_max_pos) {
    const int half = head_dim / 2;
    if (scale <= 1.0f) {
        // Below the trained context: pass-through frequencies.
        for (int i = 0; i < half; ++i) {
            const float exponent =
                static_cast<float>(2 * i) / static_cast<float>(head_dim);
            inv_freq_out[i] = 1.0f / std::pow(theta_base, exponent);
        }
        return;
    }

    for (int i = 0; i < half; ++i) {
        const float exponent =
            static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float base_inv = 1.0f / std::pow(theta_base, exponent);
        const float linear_inv = base_inv / scale;

        const float wl = wavelengths_in_orig(base_inv, orig_max_pos);
        // Ramp factor: wl >= beta_fast → keep extrapolation (base_inv);
        //              wl <= beta_slow → fully linear (linear_inv);
        //              between → linear blend.
        float t;
        if (wl >= beta_fast) {
            t = 0.0f;
        } else if (wl <= beta_slow) {
            t = 1.0f;
        } else {
            t = (beta_fast - wl) / (beta_fast - beta_slow);
            t = std::min(1.0f, std::max(0.0f, t));
        }
        inv_freq_out[i] = (1.0f - t) * base_inv + t * linear_inv;
    }
}

void alibi_slopes(float* slopes_out, int n_heads) {
    if (n_heads <= 0) return;
    auto power_of_two_slopes = [](float* out, int n) {
        const float start = std::pow(2.0f, -std::pow(2.0f, -(std::log2(static_cast<float>(n)) - 3.0f)));
        const float ratio = start;
        out[0] = start;
        for (int i = 1; i < n; ++i) out[i] = out[i - 1] * ratio;
    };

    // The canonical ALiBi rule applies the closed form when n_heads is a
    // power of two. For non-power-of-two head counts, the original paper
    // interpolates; we reuse the next-lower power of two and pad.
    auto is_power_of_two = [](int n) { return n > 0 && (n & (n - 1)) == 0; };
    if (is_power_of_two(n_heads)) {
        power_of_two_slopes(slopes_out, n_heads);
        return;
    }
    int closest = 1;
    while (closest * 2 <= n_heads) closest *= 2;
    power_of_two_slopes(slopes_out, closest);
    // Fill the remainder by sampling every other slope from the
    // next-larger power of two, matching the reference behaviour.
    const int extra = n_heads - closest;
    if (extra > 0) {
        const int bigger = closest * 2;
        std::vector<float> big(static_cast<size_t>(bigger));
        power_of_two_slopes(big.data(), bigger);
        for (int i = 0; i < extra; ++i) {
            slopes_out[closest + i] = big[static_cast<size_t>(2 * i + 1)];
        }
    }
}

void alibi_bias(float* bias_out, const float* slopes, int n_heads,
                int seq_q, int seq_k, bool causal) {
    const int pos_shift = seq_k - seq_q;
    for (int h = 0; h < n_heads; ++h) {
        const float slope = slopes[h];
        for (int i = 0; i < seq_q; ++i) {
            const int q_pos = pos_shift + i;
            for (int j = 0; j < seq_k; ++j) {
                const int delta = q_pos - j;
                const float clamped =
                    causal ? static_cast<float>(std::max(0, delta))
                           : static_cast<float>(std::abs(delta));
                bias_out[((static_cast<int64_t>(h) * seq_q + i) * seq_k) + j] =
                    -slope * clamped;
            }
        }
    }
}

}  // namespace lh
