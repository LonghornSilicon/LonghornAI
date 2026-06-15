# Mixture of Experts (`src/kernels/moe.{hpp,cpp}`)

## What it computes
- `moe_router` — `Linear(hidden → n_experts) + softmax + top-k`,
  with optional gate re-normalisation.
- `moe_dispatch` — counting-sort scatter of token activations into
  expert-major order.
- `moe_expert_mlp` — per-expert SwiGLU MLP (W_gate / W_up / SwiGLU /
  W_down), one grouped GEMM per projection.
- `moe_combine` — gate-weighted gather back to per-token outputs in a
  single shuffle pass.

## Reference path
`moe_forward` (full block) validated against a naive per-token
reference that loops over each token's top-k experts.

## Memory traffic
The MoE block **explodes** weight memory: `n_experts × (W_gate + W_up +
W_down)`. Per token only `top_k` experts execute, so per-token compute
stays roughly constant — making the block much more memory-bound than
a dense MLP at equivalent active compute.

## Arithmetic intensity
Per expert: same AI as dense GEMM, but the *block-level* AI drops by
`n_experts / top_k`. **Decode is the bandwidth crisis of MoE.**

## Roofline class
- Routing: memory-bound (a small Linear + softmax + top-k).
- Dispatch / combine: **bandwidth-limited** shuffle.
- Expert MLP: same as dense MLP per expert. Block-level performance is
  capped by the expert-weight read.

## Engine assignment
- Router: **Tensor Unit (small)** + **Softmax + Top-K Engines**.
- Dispatch / Combine: **Permutation Engine**. A token-shuffle path
  with counting-sort offsets is the silicon-relevant primitive.
- Expert MLP: **Tensor Unit** with **Grouped-GEMM dispatch** so a list
  of `(M_g, N_g, K_g)` tuples runs back-to-back without host
  intervention.

## SRAM working set
- Per-expert: GEMM tile (~32 KiB).
- Dispatch buffer: `T · top_k · hidden_dim · 4` bytes at FP32. For
  T=64, top_k=2, hidden=4096: 2 MiB. Resident across the block.
- Per-expert offsets: `4 · (n_experts + 1)` bytes. Trivial.

## DMA pattern
- Dispatch: **counting-sort scatter** — one pass over T tokens.
- Combine: **weighted gather** — one pass over `T · top_k` slots.
- Expert weights: **strided streaming**, but with a different expert
  per group → poor temporal reuse. Quantization (W4A16) is the only
  way to make this affordable.

## Datapath dtype requirements
- Router weights: typically FP16 / BF16 (small).
- Expert weights: **must be quantized** (W4A16 / W8A8) for MoE to
  ship — the weight bandwidth otherwise dominates.
- All quant scopes pass through Tensor Unit unchanged.

## Open silicon questions
- Expert-parallel sharding policy: many-expert models (DeepSeek-MoE)
  shard experts across compute tiles, requiring all-to-all. The NoC
  bandwidth derived from this is the Phase 9 driver.
- Shared-expert (DeepSeek) fast-path: an extra unconditional dense
  pass added to the combine output; potential dedicated lane.
