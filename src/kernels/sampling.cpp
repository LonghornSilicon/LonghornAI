#include "kernels/sampling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace lh {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// In-place stable softmax on a row of `vocab` logits. Returns the row max.
float stable_softmax_inplace(float* logits, int vocab) {
    float m = kNegInf;
    for (int i = 0; i < vocab; ++i) if (logits[i] > m) m = logits[i];
    if (!std::isfinite(m)) {
        // All -inf: nothing to sample. Caller should treat this as an
        // error; we return a uniform-over-survivors-of-zero, which is
        // formally undefined but stable enough for tests.
        for (int i = 0; i < vocab; ++i) logits[i] = 0.0f;
        return 0.0f;
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        const float p = std::exp(logits[i] - m);
        logits[i] = p;
        sum += p;
    }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (int i = 0; i < vocab; ++i) logits[i] *= inv;
    }
    return m;
}

}  // namespace

void apply_temperature(float* logits, int vocab, float T) {
    assert(T > 0.0f);
    const float inv = 1.0f / T;
    for (int i = 0; i < vocab; ++i) logits[i] *= inv;
}

void apply_top_k(float* logits, int vocab, int k) {
    if (k <= 0 || k >= vocab) return;
    // Find the k-th largest logit via partial sort on indices.
    std::vector<int32_t> idx(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    // The pivot value: the k-th largest logit. Anything strictly less
    // gets masked. Tied values at the boundary are kept (matches the HF
    // top-k semantics where ties pad the kept set).
    const float pivot = logits[idx[k - 1]];
    for (int i = 0; i < vocab; ++i) {
        if (logits[i] < pivot) logits[i] = kNegInf;
    }
}

void apply_top_p(float* logits, int vocab, float p) {
    if (p >= 1.0f) return;
    if (p <= 0.0f) {
        // Degenerate: keep the argmax only.
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        for (int i = 0; i < vocab; ++i)
            if (i != best) logits[i] = kNegInf;
        return;
    }
    // Sort indices by descending logit.
    std::vector<int32_t> idx(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });

    // Compute softmax probabilities in sorted order.
    float m = logits[idx[0]];
    std::vector<float> probs(static_cast<size_t>(vocab));
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(logits[idx[i]] - m);
        sum += probs[static_cast<size_t>(i)];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < vocab; ++i) probs[static_cast<size_t>(i)] *= inv;

    // Walk until cumulative probability >= p; cut off the tail.
    float cum = 0.0f;
    int cutoff = vocab - 1;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[static_cast<size_t>(i)];
        if (cum >= p) {
            cutoff = i;
            break;
        }
    }
    for (int i = cutoff + 1; i < vocab; ++i) {
        logits[idx[i]] = kNegInf;
    }
}

void apply_min_p(float* logits, int vocab, float p_min) {
    if (p_min <= 0.0f) return;
    // Compute softmax (read-only) to identify the max prob; threshold is
    // p_min * max_prob. We keep the row's logits intact for tokens that
    // pass; rejected ones go to -inf.
    float m = kNegInf;
    for (int i = 0; i < vocab; ++i) if (logits[i] > m) m = logits[i];
    if (!std::isfinite(m)) return;
    // The max probability after softmax is exp(0) / sum = 1 / sum, where
    // sum = sum(exp(logit - m)). Equivalently: prob_i / max_prob =
    // exp(logit_i - m). So the threshold prob_i >= p_min * max_prob
    // simplifies to logit_i - m >= log(p_min).
    const float thresh = std::log(p_min);
    for (int i = 0; i < vocab; ++i) {
        if (logits[i] - m < thresh) logits[i] = kNegInf;
    }
}

void apply_typical_p(float* logits, int vocab, float p) {
    if (p >= 1.0f) return;
    // Locally-typical sampling: rank tokens by |H - (-log p_i)|, the
    // surprisal-vs-entropy gap, and keep the smallest set whose mass
    // covers p. Work in log space until the final mask step.
    float m = kNegInf;
    for (int i = 0; i < vocab; ++i) if (logits[i] > m) m = logits[i];
    std::vector<float> log_probs(static_cast<size_t>(vocab));
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        const float e = std::exp(logits[i] - m);
        sum += e;
    }
    const float log_sum = std::log(sum) + m;
    for (int i = 0; i < vocab; ++i) {
        log_probs[static_cast<size_t>(i)] = logits[i] - log_sum;
    }
    // Entropy: H = -sum p_i log p_i.
    float H = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        const float lp = log_probs[static_cast<size_t>(i)];
        const float pp = std::exp(lp);
        H -= pp * lp;
    }
    // Sort by ascending |H - (-log p_i)|.
    std::vector<int32_t> idx(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float da = std::fabs(H + log_probs[static_cast<size_t>(a)]);
        const float db = std::fabs(H + log_probs[static_cast<size_t>(b)]);
        return da < db;
    });
    float cum = 0.0f;
    std::vector<bool> keep(static_cast<size_t>(vocab), false);
    for (int rank = 0; rank < vocab; ++rank) {
        const int i = idx[static_cast<size_t>(rank)];
        const float pp = std::exp(log_probs[static_cast<size_t>(i)]);
        keep[static_cast<size_t>(i)] = true;
        cum += pp;
        if (cum >= p) break;
    }
    for (int i = 0; i < vocab; ++i) {
        if (!keep[static_cast<size_t>(i)]) logits[i] = kNegInf;
    }
}

int32_t argmax_sample(const float* logits, int vocab) {
    int32_t best = 0;
    float bv = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

int32_t softmax_sample(const float* logits, int vocab, uint64_t* rng_state) {
    // Compute probabilities in a scratch buffer; sample by inverse CDF.
    std::vector<float> probs(static_cast<size_t>(vocab));
    std::memcpy(probs.data(), logits,
                static_cast<size_t>(vocab) * sizeof(float));
    stable_softmax_inplace(probs.data(), vocab);
    const double u = uniform01(rng_state);
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[static_cast<size_t>(i)];
        if (u < cum) return i;
    }
    return vocab - 1;  // fallback for fp drift
}

int32_t sample(const float* logits_in, int vocab, const SamplingPolicy& p,
               uint64_t* rng_state, float* scratch) {
    if (p.temperature <= 0.0f) {
        return argmax_sample(logits_in, vocab);
    }
    std::memcpy(scratch, logits_in, static_cast<size_t>(vocab) * sizeof(float));
    apply_temperature(scratch, vocab, p.temperature);
    if (p.top_k > 0) apply_top_k(scratch, vocab, p.top_k);
    if (p.top_p < 1.0f) apply_top_p(scratch, vocab, p.top_p);
    if (p.min_p > 0.0f) apply_min_p(scratch, vocab, p.min_p);
    if (p.typical_p < 1.0f) apply_typical_p(scratch, vocab, p.typical_p);
    return softmax_sample(scratch, vocab, rng_state);
}

}  // namespace lh
