#include "kernels/rope.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace lh {

namespace {
inline float pair_freq(int i, int head_dim, float theta_base) {
    // i indexes the rotated pair in [0, head_dim/2).
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    return 1.0f / std::pow(theta_base, exponent);
}
}  // namespace

void rope_ref(float* x, int seq, int n_heads, int head_dim,
              float theta_base, float freq_scale, bool interleaved,
              int pos_offset) {
    const int half = head_dim / 2;
    for (int s = 0; s < seq; ++s) {
        const float pos = static_cast<float>(pos_offset + s) * freq_scale;
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + ((static_cast<int64_t>(s) * n_heads + h) * head_dim);
            for (int i = 0; i < half; ++i) {
                const float angle = pos * pair_freq(i, head_dim, theta_base);
                const float c = std::cos(angle);
                const float sn = std::sin(angle);
                if (interleaved) {
                    const float a = v[2 * i];
                    const float b = v[2 * i + 1];
                    v[2 * i] = a * c - b * sn;
                    v[2 * i + 1] = a * sn + b * c;
                } else {
                    const float a = v[i];
                    const float b = v[i + half];
                    v[i] = a * c - b * sn;
                    v[i + half] = b * c + a * sn;
                }
            }
        }
    }
}

// Precompute the per-pair cos/sin for each position once and reuse across
// heads, which is where the per-element transcendental cost otherwise lands.
void rope(float* x, int seq, int n_heads, int head_dim, float theta_base,
          float freq_scale, bool interleaved, int pos_offset) {
    const int half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i) inv_freq[i] = pair_freq(i, head_dim, theta_base);

    std::vector<float> cos_tab(static_cast<size_t>(seq) * half);
    std::vector<float> sin_tab(static_cast<size_t>(seq) * half);
    for (int s = 0; s < seq; ++s) {
        const float pos = static_cast<float>(pos_offset + s) * freq_scale;
        for (int i = 0; i < half; ++i) {
            const float angle = pos * inv_freq[i];
            cos_tab[static_cast<size_t>(s) * half + i] = std::cos(angle);
            sin_tab[static_cast<size_t>(s) * half + i] = std::sin(angle);
        }
    }

    for (int s = 0; s < seq; ++s) {
        const float* cs = cos_tab.data() + static_cast<size_t>(s) * half;
        const float* sn = sin_tab.data() + static_cast<size_t>(s) * half;
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + ((static_cast<int64_t>(s) * n_heads + h) * head_dim);
            for (int i = 0; i < half; ++i) {
                const float c = cs[i];
                const float si = sn[i];
                if (interleaved) {
                    const float a = v[2 * i];
                    const float b = v[2 * i + 1];
                    v[2 * i] = a * c - b * si;
                    v[2 * i + 1] = a * si + b * c;
                } else {
                    const float a = v[i];
                    const float b = v[i + half];
                    v[i] = a * c - b * si;
                    v[i + half] = b * c + a * si;
                }
            }
        }
    }
}

}  // namespace lh
