// LonghornAI — embedding lookup (gather).
#ifndef LONGHORNAI_KERNELS_EMBEDDING_HPP
#define LONGHORNAI_KERNELS_EMBEDDING_HPP

#include <cstdint>

namespace lh {

// Gather rows from table[vocab, dim] indexed by ids[n_ids] into out[n_ids, dim].
// Each gathered row is multiplied by `scale` (1.0 for none; some models scale
// embeddings by sqrt(dim)). Out-of-range ids produce a zero row.
void embedding(const float* table, const int32_t* ids, float* out,
               int n_ids, int vocab, int dim, float scale = 1.0f);

}  // namespace lh

#endif  // LONGHORNAI_KERNELS_EMBEDDING_HPP
