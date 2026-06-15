# Kernel → Engine → SRAM → DMA → Dtype Map

The Phase 8 acceptance artifact. One row per kernel in the LonghornAI
inventory; the columns are the dimensions that drive Phase 9's
microarchitecture synthesis.

**SRAM working set** is per active tile / per active (batch, head)
pair — the on-chip scratchpad needed to keep one kernel "in flight"
without spilling to HBM. **Bound** is from `bench_main --roofline`
where measurable; otherwise estimated from arithmetic-intensity
analysis.

| Kernel | Engine | SRAM working set | DMA pattern | Primary dtype(s) | Accumulator | Bound |
|---|---|---|---|---|---|---|
| **gemm / gemm_batched** | Tensor Unit | ~32 KiB (A panel + B panel + C tile) | strided streaming | BF16/FP16/FP8/INT8/INT4 | FP32 / INT32 | compute |
| **gemm_grouped** (MoE) | Tensor Unit + grouped-GEMM dispatch | ~32 KiB per expert | strided streaming per expert | as above | FP32 / INT32 | compute |
| **gemm_w8a8** | Tensor Unit + Rescale | as GEMM (half B panel) | strided | INT8 × INT8 | INT32 → FP32 | compute |
| **gemm_w4a16_groupwise** | Tensor Unit + INT4 input-feeder dequant | as GEMM (quarter B panel) | strided + byte-aligned unpack | INT4 weights, FP32 acts | FP32 | balanced/compute |
| **gemm_fp8_e4m3** | Tensor Unit + Rescale | as GEMM (half B panel) | strided | FP8 × FP8 | FP32 | compute |
| **layernorm / rmsnorm** | Normalization Engine | one row (~16 KiB at dim=4096) | sequential streaming | BF16/FP16, FP32 reductions | FP32 | memory |
| **softmax** | Softmax Engine | one row (~16 KiB) | sequential streaming | FP32 logits | FP32 | memory |
| **reduce_sum/max/mean / argmax / topk** | Reduction Engine | small scalar + k-heap | sequential streaming | any | FP32 / INT32 | memory |
| **activation (GELU/SiLU/SwiGLU/GeGLU/...)** | Vector Unit (GEMM epilogue) | none beyond GEMM tile | pointwise streaming | input dtype | FP32 for transcendental | memory |
| **rope** | Vector Unit (Q/K projection epilogue) | cos/sin LUT (~64 KiB for seq=128) | broadcast LUT + streaming | BF16/FP16 + FP32 trig | FP32 | memory |
| **positional (NTK/YaRN/ALiBi)** | Host or fused into Attention Engine | slope table (negligible) | broadcast slopes | FP32 | FP32 | memory |
| **embedding** | DMA Engine (scatter-gather) | touched rows in flight | scatter-gather row indirect | table dtype | n/a | memory |
| **transform (transpose/permute/concat/split)** | DMA Engine | transposed-tile (~4 KiB) | transposed-tile / strided | any | n/a | memory |
| **gather_rows / scatter_add_rows** | DMA Engine | none | scatter-gather row indirect | any | FP32 for scatter-add | memory |
| **sdpa / flash_attention / flash_decode_attention** | Attention Engine | ~30 KiB (Q tile + K tile + V tile + m,ℓ,O state) | KV streamed via KV controller | BF16/FP16 (Q/K/V), FP32 state | FP32 | balanced (prefill) / memory (decode) |
| **cross_attention** | Attention Engine | as attention | as attention | same | FP32 | balanced |
| **paged_kv_append** | KV Cache Controller (write path) | one block (~32 KiB) | indirected block writes | FP16 storage (or INT8/FP8) | n/a | memory |
| **paged_attention / paged_attention_batched** | Attention Engine + KV Controller | as attention + 2 blocks resident | indirected block-grain streaming | as attention | FP32 | memory (decode) |
| **paged_kv_append_q8 / paged_attention_q8** | KV Controller (dequant on read) + Attention Engine | as above (half pool size) | indirected + on-the-fly INT8→FP16 | INT8 storage + FP32 scale | FP32 | memory |
| **quant utilities (q8/q4/fp8 quantize/dequantize)** | Vector Unit + GEMM input-feeder | one row / group buffer | sequential streaming | FP32 → low-precision | FP32 scales | memory |
| **moe_router** | Tensor Unit (small) + Softmax + Top-K | router weight tile (small) | strided | FP16 weights | FP32 | memory |
| **moe_dispatch / moe_combine** | Permutation Engine | dispatch buffer (~MiB at batch=64) | counting-sort scatter / weighted gather | activation dtype | FP32 for combine | memory |
| **moe_expert_mlp** | Tensor Unit (grouped-GEMM dispatch) | per-expert GEMM tile (~32 KiB) | strided per expert (no temporal reuse across experts) | quantized weights (W4A16/W8A8) | FP32 | memory |
| **sampling (filters + sample)** | Reduction Engine + Vector Unit + PRNG | one logit row (~512 KiB at vocab=128k) | sequential streaming | FP32 logits | FP32 | memory |
| **speculative_verify** | Vector Unit + Reduction + PRNG | 2 probability rows | streamed from target/draft buffers | FP32 probs | FP32 | memory |
| **build_tree_attention_bias** | Host / Scheduler | ~520 KiB at n_nodes=64, n_history=2048 | broadcast to attention | FP32 mask | FP32 | memory |
| **selective_scan (Mamba S6)** | Scan Engine | per-channel state (`d_state · 4` = 64 B for d_state=16) | streaming x/B/C + state-resident | BF16/FP16 + FP32 state | FP32 | memory |
| **selective_scan_chunked (Mamba-2 SSD)** | Scan Engine carry + Tensor Unit (within-chunk LT matmul) | A_bar/B_bar tables (~1 KiB/channel) + state | streaming + carry | as Mamba | FP32 | balanced |
| **wkv (RWKV v5/v6)** | Scan Engine (log-max mode) | (a, b, p, w, u) per channel ~20 B | streaming k/v + state-resident | BF16/FP16 + FP32 log-space | FP32 | memory |
| **linear_attention** | Scan Engine (outer-product mode) | per-(batch,head) S+z (~16 KiB for d_feat=d_v=64) | streaming Q/K/V + state-resident | BF16/FP16 + FP32 state | FP32 | memory/balanced |
| **conv1d_causal_depthwise** | Vector Unit | filter table (~64 KiB) + per-layer window | broadcast filter + streaming | BF16/FP16 | FP32 | memory |

---

## Engine summary

By engine block, the kernels that target it:

| Engine | Kernels |
|---|---|
| **Tensor Unit** | gemm, gemm_batched, gemm_grouped, gemm_w8a8, gemm_w4a16_groupwise, gemm_fp8_e4m3, moe_expert_mlp, moe_router, attention score/output, SSD within-chunk LT matmul |
| **Attention Engine** | sdpa, flash_attention, flash_decode_attention, cross_attention, paged_attention(_q8) |
| **KV Cache Controller** | paged_kv_append(_q8), paged_attention(_q8) read path, dequant on read, prefetch, eviction |
| **Vector Unit** | activation (all), rope, quant utilities, sampling filters, conv1d, ALiBi-fused bias add |
| **Reduction Engine** | reduce_sum/max/mean, argmax, topk, softmax base reductions |
| **Softmax Engine** | softmax, attention score softmax (fused into Attention Engine), sampling softmax |
| **Normalization Engine** | layernorm, rmsnorm (+ optional fused RoPE epilogue) |
| **Permutation Engine** | moe_dispatch, moe_combine, transpose/permute (when too tile-shaped for DMA), gather/scatter |
| **Scan Engine** | selective_scan (Mamba S6), selective_scan_chunked (Mamba-2 SSD cross-chunk carry), wkv (RWKV v5/v6), linear_attention |
| **DMA Engine** | embedding, transform (transpose/permute/concat/split), gather_rows/scatter_add_rows, all weight loads |
| **Host / Scheduler** | NTK/YaRN setup, build_tree_attention_bias |

---

## Roofline at a glance

| Class | Kernels (representative) |
|---|---|
| **Compute-bound** | dense GEMM, W8A8/W4A16/FP8 GEMM, MoE expert MLP (per expert) |
| **Balanced** | attention (prefill), SSD within-chunk, linear_attention with wide state |
| **Memory-bound** | RMSNorm, softmax, sampling, embedding, RoPE, paged_attention (decode), MoE block (decode), wkv, conv1d, all transforms |

The **memory-bound** column is the long pole of decode performance.
Phase 9's KV Controller + on-chip-dequant decisions are the highest-
leverage knobs on the system.
