#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "kernels/reduction.hpp"
#include "test_util.hpp"

TEST(Argmax, ReturnsHighestValueIndex) {
    std::vector<float> x = {0.1f, 0.7f, 0.2f, 0.5f, 0.7f};  // tie at idx 1, 4
    int32_t idx = -1;
    float val = 0.0f;
    lh::argmax(x.data(), &idx, &val, 1, 5);
    EXPECT_EQ(idx, 1);  // lowest-index tie wins
    EXPECT_FLOAT_EQ(val, 0.7f);
}

TEST(Argmax, MultiRow) {
    std::vector<float> x = {0.1f, 0.7f, 0.2f,
                            0.9f, 0.0f, 0.5f};
    std::vector<int32_t> idx(2);
    std::vector<float> val(2);
    lh::argmax(x.data(), idx.data(), val.data(), 2, 3);
    EXPECT_EQ(idx[0], 1);
    EXPECT_EQ(idx[1], 0);
    EXPECT_FLOAT_EQ(val[0], 0.7f);
    EXPECT_FLOAT_EQ(val[1], 0.9f);
}

TEST(TopK, ReturnsSortedDescending) {
    std::vector<float> x = {0.1f, 0.7f, 0.2f, 0.5f, 0.9f, 0.3f};
    std::vector<float> v(3);
    std::vector<int32_t> idx(3);
    lh::topk(x.data(), v.data(), idx.data(), 1, 6, 3);
    EXPECT_FLOAT_EQ(v[0], 0.9f);
    EXPECT_FLOAT_EQ(v[1], 0.7f);
    EXPECT_FLOAT_EQ(v[2], 0.5f);
    EXPECT_EQ(idx[0], 4);
    EXPECT_EQ(idx[1], 1);
    EXPECT_EQ(idx[2], 3);
}

TEST(TopK, TieBreakByLowestIndex) {
    std::vector<float> x = {0.5f, 0.5f, 0.5f, 0.1f};
    std::vector<float> v(2);
    std::vector<int32_t> idx(2);
    lh::topk(x.data(), v.data(), idx.data(), 1, 4, 2);
    EXPECT_FLOAT_EQ(v[0], 0.5f);
    EXPECT_FLOAT_EQ(v[1], 0.5f);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
}

TEST(TopK, KGreaterThanDimClamps) {
    std::vector<float> x = {0.3f, 0.1f};
    std::vector<float> v(2);
    std::vector<int32_t> idx(2);
    // Asking for k=5 from dim=2: kernel should fill 2 entries and not write
    // past the end. The caller is responsible for sizing the output to k.
    lh::topk(x.data(), v.data(), idx.data(), 1, 2, 5);
    EXPECT_FLOAT_EQ(v[0], 0.3f);
    EXPECT_FLOAT_EQ(v[1], 0.1f);
}
