# Activations (`src/kernels/activation.{hpp,cpp}`)

## What it computes
Pointwise: GELU (erf and tanh variants), SiLU, Sigmoid, Tanh, ReLU.
Gated: SwiGLU `silu(a) ⊙ b`, GeGLU `gelu_tanh(a) ⊙ b` (Llama / Mistral
MLP block default).

## Reference path
`gelu_erf`, `gelu_tanh`, `silu`, `sigmoid`, `relu`, `tanh_act`,
`swiglu`, `geglu`.

## Memory traffic
Read N elements, write N. Gated variants read 2N, write N.

## Arithmetic intensity
- Linear activations (ReLU/Tanh/Sigmoid): ~1 FLOP/elem; **memory-bound.**
- GELU/SiLU: ~5–10 FLOPs/elem (transcendental); **memory-bound** at
  small sizes, **compute-bound** under fully-pipelined exp/erf.
- SwiGLU/GeGLU: same as the underlying activation, halved bandwidth
  per output (one combined load).

## Roofline class
**Memory-bound** in practice. Always best fused into the producing
GEMM's epilogue.

## Engine assignment
**Vector Unit / GEMM epilogue.** None of these warrant a dedicated
engine. SwiGLU and GeGLU motivate a **fused MatMul + gate +
activation** epilogue path because gate + up + activation + down is the
hottest fused pattern in Llama-class MLPs.

## SRAM working set
None beyond the GEMM tile.

## DMA pattern
**Pointwise streaming.** Inherits from the upstream tile.

## Datapath dtype requirements
- Input: any (BF16 default; FP32 internally for the transcendental).
- Output: input dtype.

## Open silicon questions
- Which exp/erf approximation (Taylor degree, range reduction) hits the
  area / accuracy sweet spot.
- Whether to expose ReLU as a separate fast path or fold it into a
  generic activation lane.
