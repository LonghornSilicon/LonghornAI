#include "kernels/embedding.hpp"

namespace lh {

void embedding(const float* table, const int32_t* ids, float* out,
               int n_ids, int vocab, int dim, float scale) {
    for (int t = 0; t < n_ids; ++t) {
        float* orow = out + static_cast<int64_t>(t) * dim;
        const int32_t id = ids[t];
        if (id < 0 || id >= vocab) {
            for (int i = 0; i < dim; ++i) orow[i] = 0.0f;
            continue;
        }
        const float* erow = table + static_cast<int64_t>(id) * dim;
        for (int i = 0; i < dim; ++i) orow[i] = erow[i] * scale;
    }
}

}  // namespace lh
