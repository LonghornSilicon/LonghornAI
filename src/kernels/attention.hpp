// LonghornAI — attention kernels.
//
// Tensor layout (row-major):
//   Q: [batch, n_q_heads,  seq_q, head_dim]
//   K: [batch, n_kv_heads, seq_k, head_dim]
//   V: [batch, n_kv_heads, seq_k, head_dim]
//   O: [batch, n_q_heads,  seq_q, head_dim]
//
// Head configurations are expressed purely through n_kv_heads:
//   MHA: n_kv_heads == n_q_heads
//   MQA: n_kv_heads == 1
//   GQA: n_q_heads % n_kv_heads == 0   (each kv head serves a contiguous group)
//
// Causal masking uses aligned positions: query i attends to keys
// j <= (seq_k - seq_q) + i, which is correct for both prefill (seq_q == seq_k)
// and single/multi-token decode (seq_q < seq_k).
#ifndef LONGHORNAI_KERNELS_ATTENTION_HPP
#define LONGHORNAI_KERNELS_ATTENTION_HPP

namespace lh {

struct AttnConfig {
    int batch = 1;
    int n_q_heads = 1;
    int n_kv_heads = 1;
    int seq_q = 1;
    int seq_k = 1;
    int head_dim = 1;
    bool causal = false;
    // Softmax scale. If <= 0, defaults to 1/sqrt(head_dim) at call time.
    float scale = 0.0f;
};

// Reference scaled dot-product attention; the correctness anchor.
void sdpa(const float* Q, const float* K, const float* V, float* O,
          const AttnConfig& cfg);

// FlashAttention-style tiled attention with online-softmax rescaling. No
// score matrix is materialized; must match `sdpa` within tolerance.
void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg, int block_k = 32);

// Append `seq_new` timesteps of K/V into per-head caches laid out as
// [n_kv_heads, max_seq, head_dim], writing at position `past_len`.
//   k_new / v_new: [n_kv_heads, seq_new, head_dim]
void kv_cache_append(float* k_cache, float* v_cache,
                     const float* k_new, const float* v_new,
                     int n_kv_heads, int head_dim, int max_seq,
                     int past_len, int seq_new);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_ATTENTION_HPP
