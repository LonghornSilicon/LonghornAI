# 1D causal conv (`src/kernels/conv1d.{hpp,cpp}`)

## What it computes
`y[b, t, c] = bias[c] + Σ_{k=0..K-1} x[b, t-k, c] · weight[c, k]`,
zero-pad on the left. Depthwise (no cross-channel mixing). Used as
Mamba's input conv.

## Reference path
`conv1d_causal_depthwise`. Single-pass direct implementation.

## Memory traffic
- Reads: `K · channels` weights (broadcast) + `seq · channels` inputs.
- Writes: `seq · channels` outputs.
- Filter length K is typically 4 in Mamba.

## Arithmetic intensity
`K` FMAs per output = ~K FLOPs/elem. For K=4: ~4 FLOPs/elem.
**Memory-bound.**

## Roofline class
**Memory-bound.** A 4-tap FIR per channel; the silicon work is
trivial relative to the activation read.

## Engine assignment
**Vector Unit.** Sits alongside the Scan Engine in the Mamba block
pipeline (this conv runs before the selective scan). No dedicated
engine warranted.

## SRAM working set
- Filter table: `channels · K · 4` bytes. For 4096 channels × K=4:
  64 KiB. Resident for the layer.
- Sliding window: `(K-1) · channels · 4` bytes per active layer.
  Tiny.

## DMA pattern
- Weights: **broadcast**, loaded once per layer.
- Inputs / outputs: **sequential streaming**.

## Datapath dtype requirements
- Inputs / weights: BF16 / FP16.
- Accumulator: **FP32**.

## Open silicon questions
- None significant — this is a small, well-understood operator.
