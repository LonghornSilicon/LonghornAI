# RWKV WKV (`src/kernels/rwkv.{hpp,cpp}`)

## What it computes
RWKV v5/v6 numerically stable WKV recurrence:
```
y[t] = (Σ_{i<t} exp(-(t-i)·w + k_i) · v_i + exp(u + k_t) · v_t) /
       (Σ_{i<t} exp(-(t-i)·w + k_i) + exp(u + k_t))
```
Per (batch, channel). State `(a, b, p)` per channel, with `p` the
running shifted log-max for stability.

## Reference path
`wkv` (recurrent). Validated against an O(L²) naive reference for
short sequences and stress-tested for finiteness on `L=2048` with
key magnitudes that would overflow naive `exp()`.

## Memory traffic
Per step per channel:
- Reads: `k[t]`, `v[t]` (2 floats).
- Constants: `w[c]`, `u[c]` (loaded once per row).
- Writes: `y[t]` (1 float).
- State `(a, b, p)`: 12 bytes per channel, register-resident.

## Arithmetic intensity
~10 FLOPs per step per channel (two log-space combines, ratio, decay).
**Memory-bound** at the channel level; trivially parallel across
channels.

## Roofline class
**Memory-bound** in absolute terms, but the per-channel state is so
small that channel parallelism easily saturates DRAM bandwidth.

## Engine assignment
**Scan Engine.** Same shape as Mamba's per-channel state recurrence;
the same Scan Engine instance can run both architectures. The numeric
specialisation (log-max bookkeeping) is what differs and can be a
microcode mode.

## SRAM working set
Per channel: `(a, b, p, w, u)` ≈ 20 bytes register-resident. Across
n_channels: trivial.

## DMA pattern
- k, v: **sequential streaming**.
- w, u: **broadcast**, loaded once per row.
- y: **sequential streaming**.

## Datapath dtype requirements
- Inputs: BF16 / FP16.
- State and log-space lanes: **FP32 hard requirement** (the running
  max bookkeeping breaks otherwise).
- Exp/log lanes: shared with the softmax engine's exp pipeline.

## Open silicon questions
- Whether RWKV v7's state-matrix W form (a `d_state × d_state` matrix
  per channel) deserves a separate kernel or whether the same Scan
  Engine handles it via a tiled multiply mode.
