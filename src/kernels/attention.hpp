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
//
// Optional masks compose with `causal`:
//   - `bias`: additive `[seq_q, seq_k]` or `[n_q_heads, seq_q, seq_k]` mask
//     applied *before* softmax. -inf entries hard-mask, finite entries bias
//     scores (e.g. ALiBi). Selected by `bias_per_head`.
//   - `window`: if > 0, adds a sliding-window constraint
//     j >= (seq_k - seq_q) + i - window + 1.
//   - `chunk`: if > 0, restricts attention to within chunks of size `chunk`
//     (token i can only attend to keys in the same chunk).
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

    // Optional additive bias on attention scores. If `bias_per_head` is true,
    // shape is [n_q_heads, seq_q, seq_k]; otherwise [seq_q, seq_k]. Pass
    // `nullptr` for no bias. ALiBi populates the per-head form; tree masks
    // and arbitrary structured masks pass either form.
    const float* bias = nullptr;
    bool bias_per_head = false;

    // Sliding-window radius (Mistral-style). 0 disables the window.
    int window = 0;

    // Chunk size for chunked attention. 0 disables chunking.
    int chunk = 0;
};

// Reference scaled dot-product attention; the correctness anchor.
void sdpa(const float* Q, const float* K, const float* V, float* O,
          const AttnConfig& cfg);

// FlashAttention-style tiled attention with online-softmax rescaling. No
// score matrix is materialized; must match `sdpa` within tolerance.
void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg, int block_k = 32);

// FlashDecoding: split the K/V dimension across `splits` partitions, run
// flash-style attention on each, then merge the partial (m, l, O) states.
// Designed for low-batch decode (seq_q small, seq_k large) where a single
// query has so much work that splitting the K dimension is the right
// parallelism axis. With `splits == 1` it reduces to `flash_attention`.
void flash_decode_attention(const float* Q, const float* K, const float* V,
                            float* O, const AttnConfig& cfg,
                            int block_k = 32, int splits = 4);

// Explicit cross-attention: same math as SDPA but with `seq_q` queries
// attending over `seq_k` keys/values from a different stream. Causal masking
// is rejected here by convention (cross-attention is acausal); pass a custom
// `bias` for any structured constraint.
void cross_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttnConfig& cfg);

// Append `seq_new` timesteps of K/V into per-head caches laid out as
// [n_kv_heads, max_seq, head_dim], writing at position `past_len`.
//   k_new / v_new: [n_kv_heads, seq_new, head_dim]
void kv_cache_append(float* k_cache, float* v_cache,
                     const float* k_new, const float* v_new,
                     int n_kv_heads, int head_dim, int max_seq,
                     int past_len, int seq_new);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_ATTENTION_HPP
