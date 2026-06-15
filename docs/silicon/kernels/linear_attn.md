# Linear attention (`src/kernels/linear_attn.{hpp,cpp}`)

## What it computes
```
S[t] = Σ_{i ≤ t} φ(K_i) ⊗ V_i          ∈ R^{d_feat × d_v}
y[t] = (φ(Q_t)ᵀ @ S[t]) / (φ(Q_t)ᵀ @ Σ_{i ≤ t} φ(K_i))
```
Caller applies the feature map φ. `normalize=false` returns the bare
numerator (RetNet retention shape).

## Reference path
`linear_attention`. Validated against an independent rank-1 step-by-
step reference.

## Memory traffic
Per step per (batch, head):
- Reads: `Q[t]` (`d_feat`), `K[t]` (`d_feat`), `V[t]` (`d_v`).
- Writes: `y[t]` (`d_v`).
- State `S` (`d_feat · d_v`) and `z` (`d_feat`) register-resident.

## Arithmetic intensity
Per step per head: a rank-1 update (`d_feat · d_v`) + an output
project (`d_feat · d_v`). For d_feat=d_v=64 that's ~8k FLOPs per
step against ~2 KiB of fresh K/V/Q reads → AI ≈ 4. **Memory-bound**
to balanced.

## Roofline class
**Memory-bound** for small heads; pushes balanced for wider state.

## Engine assignment
**Scan Engine.** Shape: per-(batch, head) state update + readout. Same
engine as Mamba SSM and RWKV WKV with the numerical specialisation
(rank-1 outer-product update) as a microcode mode.

## SRAM working set
Per (batch, head): `S` = `d_feat · d_v · 4` bytes + `z` = `d_feat · 4`.
For d_feat=d_v=64: 16 KiB per head. Held in scratchpad while a head's
sequence is in flight.

## DMA pattern
- Q, K, V: **sequential streaming**.
- S, z: **register-resident** between steps; spilled per-head on a
  context switch.
- y: **sequential streaming**.

## Datapath dtype requirements
- Q, K, V: BF16 / FP16.
- State `S`, `z`: **FP32** to avoid drift over long sequences.
- Final ratio: FP32.

## Open silicon questions
- Whether to expose a separate "RetNet retention" mode that adds the
  per-step exponential decay against `S` (just multiplies state by a
  decay factor each step).
