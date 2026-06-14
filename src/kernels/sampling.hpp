// LonghornAI — sampling kernels.
//
// Logit-space filters and a multinomial sampler. The pipeline is the one
// every modern serving stack ships: starting with raw logits, apply
// temperature, top-k, top-p, min-p (in any combination), then softmax and
// draw. Each filter operates in place on a row of `vocab` logits and
// expresses "rejected" tokens by writing -inf into their slot, leaving
// downstream softmax to mass-zero them.
//
// RNG: a small splitmix64 state. Deterministic across platforms (the
// `rng_state` is a uint64_t the caller seeds and threads through). The
// PRNG is not cryptographic; it's a fast, well-distributed generator
// suitable for sampling thousands of tokens per second.
#ifndef LONGHORNAI_KERNELS_SAMPLING_HPP
#define LONGHORNAI_KERNELS_SAMPLING_HPP

#include <cstdint>

namespace lh {

struct SamplingPolicy {
    float temperature = 1.0f;  // 0 -> greedy/argmax (top_k/top_p ignored)
    int top_k = 0;             // 0 -> disabled
    float top_p = 1.0f;        // 1.0 -> disabled (probability mass cap)
    float min_p = 0.0f;        // 0.0 -> disabled (relative to top prob)
    float typical_p = 1.0f;    // 1.0 -> disabled (locally-typical sampling)
};

// --- splitmix64 -----------------------------------------------------------

inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform double in [0, 1). Top 53 bits → mantissa.
inline double uniform01(uint64_t* state) {
    return (splitmix64_next(state) >> 11) *
           (1.0 / 9007199254740992.0);  // 2^53
}

// --- in-place logit filters ----------------------------------------------

// Divide logits by temperature. T <= 0 is treated as a request for greedy;
// the caller should branch to argmax in that case rather than calling
// this. The kernel asserts T > 0.
void apply_temperature(float* logits, int vocab, float T);

// Keep only the K largest logits (others to -inf). K >= vocab is a no-op.
void apply_top_k(float* logits, int vocab, int k);

// Keep the smallest set whose softmax probability mass is >= p; the rest
// to -inf. p in (0, 1]; p == 1 is a no-op. Tokens are ranked by logit;
// the cutoff is inclusive of the first token that pushes the mass over p
// (matches the HF default).
void apply_top_p(float* logits, int vocab, float p);

// Keep tokens whose softmax probability is at least p_min times the max
// probability. Cheaper than top-p (no sort) and bounds the worst-case
// retained set by the head's confidence.
void apply_min_p(float* logits, int vocab, float p_min);

// Locally-typical sampling: keep tokens whose surprisal is closest to the
// distribution entropy until the cumulative mass reaches p (Meister 2022).
void apply_typical_p(float* logits, int vocab, float p);

// --- sampling ------------------------------------------------------------

// Argmax (greedy). Lowest-index tie-break.
int32_t argmax_sample(const float* logits, int vocab);

// Multinomial sample from `logits` after applying stable softmax. Mutates
// `rng_state`.
int32_t softmax_sample(const float* logits, int vocab, uint64_t* rng_state);

// Convenience: full pipeline.
//   - temperature == 0 -> greedy (returns argmax)
//   - else: temperature -> top_k -> top_p -> min_p -> typical_p ->
//           softmax_sample
int32_t sample(const float* logits_in, int vocab, const SamplingPolicy& p,
               uint64_t* rng_state, float* scratch /* len vocab */);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_SAMPLING_HPP
