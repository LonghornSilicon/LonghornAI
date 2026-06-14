#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/kvcache.hpp"
#include "kernels/kvcache_quant.hpp"
#include "kernels/paged_attention.hpp"
#include "test_util.hpp"

namespace {

double mean_sq_err(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        s += d * d;
    }
    return s / a.size();
}

}  // namespace

TEST(KvQuant, Q8AppendRoundTripsWithinTolerance) {
    constexpr int kvh = 2, S = 9, D = 8;
    constexpr int block_size = 4;
    auto K = lh_test::random_vector(kvh * S * D, 2200);
    auto V = lh_test::random_vector(kvh * S * D, 2201);

    lh::PagedCacheLayout L;
    L.num_blocks = 4;
    L.block_size = block_size;
    L.n_kv_heads = kvh;
    L.head_dim = D;
    const int64_t pool_n =
        static_cast<int64_t>(L.num_blocks) * kvh * block_size * D;
    const int64_t scale_n =
        static_cast<int64_t>(L.num_blocks) * kvh * block_size;
    std::vector<int8_t> Kp(pool_n, 0);
    std::vector<int8_t> Vp(pool_n, 0);
    std::vector<float> Ks(scale_n, 0.0f);
    std::vector<float> Vs(scale_n, 0.0f);

    const int n_blocks = (S + block_size - 1) / block_size;
    std::vector<int32_t> table(n_blocks);
    for (int i = 0; i < n_blocks; ++i) table[i] = i;
    lh::paged_kv_append_q8(Kp.data(), Vp.data(), Ks.data(), Vs.data(),
                           K.data(), V.data(), table.data(), L, 0, S);

    // Round-trip dequantize and check the per-row max error is bounded by
    // the quantization step (scale / 2 in expectation). We check max abs
    // error < scale + a slop.
    for (int h = 0; h < kvh; ++h) {
        for (int s = 0; s < S; ++s) {
            const int phys = table[s / block_size];
            const int slot = s % block_size;
            const float ks =
                Ks[lh::paged_scale_offset(L, phys, h, slot)];
            const int64_t pool_off = lh::paged_offset(L, phys, h, slot, 0);
            for (int t = 0; t < D; ++t) {
                const float dq = Kp[pool_off + t] * ks;
                const float ref = K[((h * S + s) * D) + t];
                EXPECT_LT(std::fabs(dq - ref), ks + 1e-6f);
            }
        }
    }
}

TEST(KvQuant, Q8AttentionMatchesFp32WithinTolerance) {
    // INT8 KV is lossy; we verify the q8 path is within a documented
    // tolerance of the fp32 path on the same data.
    constexpr int qh = 4, kvh = 2, sq = 1, sk = 32, d = 16;
    constexpr int block_size = 8;

    auto Q = lh_test::random_vector(qh * sq * d, 2300);
    auto K = lh_test::random_vector(kvh * sk * d, 2301);
    auto V = lh_test::random_vector(kvh * sk * d, 2302);

    lh::PagedCacheLayout L;
    L.num_blocks = 8;
    L.block_size = block_size;
    L.n_kv_heads = kvh;
    L.head_dim = d;
    const int needed = (sk + block_size - 1) / block_size;
    std::vector<int32_t> table(needed);
    for (int i = 0; i < needed; ++i) table[i] = i;

    // FP32 paged baseline.
    std::vector<float> Kp_f32(L.num_blocks * kvh * block_size * d, 0.0f);
    std::vector<float> Vp_f32(L.num_blocks * kvh * block_size * d, 0.0f);
    lh::paged_kv_append(Kp_f32.data(), Vp_f32.data(), K.data(), V.data(),
                        table.data(), L, 0, sk);

    lh::PagedAttnConfig pcfg;
    pcfg.n_q_heads = qh;
    pcfg.n_kv_heads = kvh;
    pcfg.seq_q = sq;
    pcfg.head_dim = d;
    pcfg.causal = true;

    std::vector<float> O_f32(qh * sq * d, 0.0f);
    lh::paged_attention(Q.data(), Kp_f32.data(), Vp_f32.data(), table.data(),
                        sk, O_f32.data(), pcfg, L);

    // Q8 path.
    std::vector<int8_t> Kp_q8(L.num_blocks * kvh * block_size * d, 0);
    std::vector<int8_t> Vp_q8(L.num_blocks * kvh * block_size * d, 0);
    std::vector<float> Ks(L.num_blocks * kvh * block_size, 0.0f);
    std::vector<float> Vs(L.num_blocks * kvh * block_size, 0.0f);
    lh::paged_kv_append_q8(Kp_q8.data(), Vp_q8.data(), Ks.data(), Vs.data(),
                           K.data(), V.data(), table.data(), L, 0, sk);
    std::vector<float> O_q8(qh * sq * d, 0.0f);
    lh::paged_attention_q8(Q.data(), Kp_q8.data(), Vp_q8.data(), Ks.data(),
                           Vs.data(), table.data(), sk, O_q8.data(), pcfg, L);

    // INT8 KV with per-row scales: empirically MSE < ~1e-4 on benign
    // random inputs of unit variance; we allow 1e-3 to leave headroom for
    // pathological draws under the fixed seed.
    EXPECT_LT(mean_sq_err(O_q8, O_f32), 1e-3);
    // Per-element absolute error bound: same scale.
    for (size_t i = 0; i < O_q8.size(); ++i) {
        EXPECT_LT(std::fabs(O_q8[i] - O_f32[i]), 5e-2f)
            << "i=" << i << " q8=" << O_q8[i] << " f32=" << O_f32[i];
    }
}
