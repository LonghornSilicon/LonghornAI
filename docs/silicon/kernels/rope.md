# RoPE (`src/kernels/rope.{hpp,cpp}`)

## What it computes
Rotary position embedding: per pair `(x_a, x_b)` apply
`(c, −s; s, c)` with `s = sin(p·θ_i)`, `c = cos(p·θ_i)`,
`θ_i = base^(−2i/d)`. Two layouts: interleaved `(2i, 2i+1)` (GPT-J)
and half-rotation `(i, i+d/2)` (GPT-NeoX, Llama HF).

## Reference path
- `rope_ref` (per-element trig per position).
- `rope` (precomputed cos/sin table, reused across heads).

## Memory traffic
- Reads: `seq · n_heads · head_dim` floats.
- Writes: same (in-place).
- Internal: small `(seq · head_dim/2)` cos/sin table.

## Arithmetic intensity
4 FLOPs per pair (two muls, two FMAs). **Memory-bound.**

## Roofline class
**Memory-bound.** Best fused into the Q/K projection epilogue so the
output never leaves the register file before rotation.

## Engine assignment
**Vector Unit / Q-K projection epilogue.** No dedicated engine. The
cos/sin table is small enough to live in a tile-local LUT.

## SRAM working set
Cos/sin table: `2 · seq · (head_dim / 2) · 4` bytes. For seq=2048,
head_dim=128: 1 MiB; for seq=128 (typical decode tile): 64 KiB.
Computed lazily per-row in the optimised path.

## DMA pattern
**Sequential streaming** of the Q/K stream; **broadcast** of the
cos/sin table (small, per-position).

## Datapath dtype requirements
- Input: BF16/FP16 typical (the projection output).
- Trig lanes: **FP32** (cos/sin precomputed in FP32 and downcast
  on use).

## Open silicon questions
- Whether NTK / YaRN scaling motivates per-head θ tables (different
  bands of frequencies per head). Currently a single-base table works.
