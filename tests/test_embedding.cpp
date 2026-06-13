#include <gtest/gtest.h>

#include <vector>

#include "kernels/embedding.hpp"
#include "test_util.hpp"

TEST(Embedding, GathersRows) {
    const int vocab = 50, dim = 16, n_ids = 20;
    auto table = lh_test::random_vector(vocab * dim, 601);
    auto ids = lh_test::random_ids(n_ids, vocab, 602);
    std::vector<float> out(n_ids * dim);
    lh::embedding(table.data(), ids.data(), out.data(), n_ids, vocab, dim);

    for (int t = 0; t < n_ids; ++t) {
        const float* erow = table.data() + ids[t] * dim;
        for (int i = 0; i < dim; ++i) {
            EXPECT_FLOAT_EQ(out[t * dim + i], erow[i]);
        }
    }
}

TEST(Embedding, ScaleAndOutOfRange) {
    const int vocab = 4, dim = 3;
    std::vector<float> table = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> ids = {2, -1, 99};
    std::vector<float> out(ids.size() * dim, -123.0f);
    lh::embedding(table.data(), ids.data(), out.data(),
                  static_cast<int>(ids.size()), vocab, dim, 2.0f);

    EXPECT_FLOAT_EQ(out[0], 14.0f);  // row 2 (7,8,9) * 2
    EXPECT_FLOAT_EQ(out[1], 16.0f);
    EXPECT_FLOAT_EQ(out[2], 18.0f);
    for (int i = 3; i < 9; ++i) EXPECT_FLOAT_EQ(out[i], 0.0f);  // OOB -> zero
}
