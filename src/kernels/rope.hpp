// LonghornAI — rotary position embeddings (RoPE).
//
// Operates in place on x laid out as [seq, n_heads, head_dim]. Supports the
// two common layouts and linear/NTK frequency scaling.
//   - interleaved: rotates pairs (2i, 2i+1)            (GPT-J / original RoPE)
//   - half-rotation: rotates pairs (i, i + head_dim/2) (GPT-NeoX / Llama HF)
// `freq_scale` multiplies the position (linear position interpolation; pass
// 1.0 for none). For NTK scaling pass an adjusted `theta_base`.
#ifndef LONGHORNAI_KERNELS_ROPE_HPP
#define LONGHORNAI_KERNELS_ROPE_HPP

namespace lh {

void rope_ref(float* x, int seq, int n_heads, int head_dim,
              float theta_base = 10000.0f, float freq_scale = 1.0f,
              bool interleaved = true, int pos_offset = 0);

void rope(float* x, int seq, int n_heads, int head_dim,
          float theta_base = 10000.0f, float freq_scale = 1.0f,
          bool interleaved = true, int pos_offset = 0);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_ROPE_HPP
