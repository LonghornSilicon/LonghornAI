# Roofline Analysis

The roofline model frames every kernel by two numbers: its **arithmetic
intensity** (AI = FLOPs per byte of memory traffic) and the **target
machine's ridge point** (peak FLOPS / peak bandwidth). Kernels with
AI ≪ ridge are memory-bound; AI ≫ ridge, compute-bound; near the
ridge, balanced.

For LonghornAI we use a *target-independent* ridge of **10 FLOPs/byte**
as the default classifier — representative of a contemporary x86 core
with ~50 GFLOPS/lane and ~5 GB/s sustained per-thread DRAM bandwidth.
The classification band is `[0.5x, 2x]` of the ridge to absorb
measurement noise:

```
              ^ achieved GFLOPS
              |
   compute    |              .  .  .  .  .  .  .  .   ← peak FLOPS
   roof       |          .
              |       .
              |     .            <- ridge ≈ 10 FLOPs/byte
              |   .  ←  AI = 5 (balanced)        AI = 50 (compute)
              | . ←  AI = 0.5 (memory)
              .
              +------------------------------------>  arithmetic intensity
```

A target accelerator with peak 1 PFLOP / peak 5 TB/s HBM has ridge
~200 FLOPs/byte — every kernel currently classified as "compute" on
the CPU baseline becomes balanced or memory-bound at silicon scale.
**The bound classes shift right as the target gets faster relative to
memory.**

## Per-kernel AI

From `./build/longhorn_bench --roofline`. Numbers are from the
representative shape in `bench_main.cpp`; the classification is per
that shape, not a universal claim about the kernel.

```
kernel                                AI    GFLOP/s  GB/s   bound
gemm/qkv_small[128x768x768]           48.00  31.04   0.65   compute
gemm/mlp_up[128x2048x768]             52.07  21.80   0.42   compute
gemm/mlp_down[128x768x2048]           52.07  31.11   0.60   compute
rmsnorm[128x4096]                      0.38   5.55  14.81   memory
softmax[128x4096]                      0.62   1.84   2.94   memory
sdpa/causal[1x8x128x64]               16.00  17.26   1.08   balanced
flash/causal[1x8x128x64]              16.00  16.22   1.01   balanced
```

`rmsnorm` sustains ~70% of single-thread DRAM bandwidth — the memory
roof is the limit, not the compute roof. `gemm` sits clearly above
the ridge. Attention is right at the ridge for the prefill shape
tested; at decode shapes (`seq_q = 1, seq_k` large) it falls into the
memory-bound class because the KV read dominates.

## Interpretation per kernel family

- **Dense GEMM**: compute-bound at production shapes. Silicon picks
  the tensor unit's `MR × NR × KR` to maximise reuse.
- **Quantized GEMM (W4A16, W8A8, FP8)**: same FLOPs, fewer bytes —
  AI rises by the byte-ratio. Critical for decode where dense GEMV
  would be memory-bound.
- **MoE expert MLP**: per expert, compute-bound like dense GEMM. At
  the *block* level, AI drops by `n_experts / top_k` because each
  expert's weights are read fresh — memory-bound unless aggressive
  weight quantization is applied.
- **Attention (prefill)**: AI ≈ head_dim / 2 (~32 for d=64) →
  balanced.
- **Attention (decode)**: KV bandwidth dominates → memory-bound.
  This is the single most important fact in inference performance.
- **Norms / softmax / sampling / RoPE / activations / transforms /
  embedding**: memory-bound. None justify a dedicated compute lane;
  the answer is fusion (normalisation engine pulls in RoPE, GEMM
  epilogue absorbs activations, sampling fuses with LM-head GEMM).
- **Mamba (reference scan)**: memory-bound at AI ~3. The SSD recipe
  pushes within-chunk work into matmul shape — that part lands on the
  tensor unit and becomes balanced.
- **RWKV WKV / linear attention**: per-channel state recurrence;
  memory-bound at the channel level, easily saturating DRAM with
  channel parallelism.

## Silicon implication: the memory roof

Roughly **two-thirds of the kernel inventory is memory-bound** at
realistic shapes. This is not a flaw — it's the actual shape of
inference workloads, and it dictates the architecture:

1. **HBM bandwidth is the single most-leveraged Phase 9 parameter.**
   Compute headroom is plentiful; bandwidth is scarce. Spend silicon
   area on more HBM channels before more MAC units.
2. **The KV Cache Controller is the single most consequential block**,
   because decode attention is memory-bound and the KV cache is the
   biggest read in decode.
3. **Quantization is a bandwidth multiplier**, not just a compression
   trick. W4A16 weights and INT8/FP8 KV both raise effective compute
   throughput by reducing the byte side of AI.
4. **Fusion wins twice**: a fused `RMSNorm + RoPE + Q/K/V projection`
   pass eliminates two round-trips through HBM. Phase 9's epilogue
   composability is worth the design effort.

The shape of the inventory is the input to Phase 9. The next document
(`docs/silicon/architecture.md`, to be written in Phase 9) translates
this into block diagrams, SRAM tier sizes, and ISA semantics.
