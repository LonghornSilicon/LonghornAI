# Selective state-space scan (`src/kernels/ssm.{hpp,cpp}`)

## What it computes
Mamba S6 selective scan:
```
discretize:  A_bar = exp(delta · A)
             B_bar = delta · B
recurrence:  h[t]  = A_bar[t] * h[t-1] + B_bar[t] * x[t]
readout:     y[t]  = C[t] @ h[t]  (+ D · x[t])
```
Per (batch, channel) over the sequence.

`selective_scan_chunked` is the Mamba-2 SSD recipe: within a chunk the
recurrence becomes a small structured matmul (semi-separable matrix);
between chunks the state carries forward. Same numerics as the
reference.

## Reference path
`selective_scan_ref` (sequential) is the golden numerics oracle.
`selective_scan_chunked` is verified to match for chunk sizes in
`{1, 2, 4, 8, seq, > seq}`.

## Memory traffic
Per step per channel:
- Reads: `x[t]` (1), `delta[t]` (1), `A` (d_state), `B[t]` (d_state),
  `C[t]` (d_state).
- Writes: `y[t]` (1).
- State `h` held in registers per channel.

Block-level: `~5 · L · D · 4` bytes input + `L · D` output.

## Arithmetic intensity
Per step per channel: ~4 FLOPs per state element. Mamba's `d_state` is
typically 16, so per token: ~64 FLOPs per channel and ~5 · 4 = 20
input bytes per channel → AI ~3. **Memory-bound** in the reference
scan.

The SSD chunked form **raises AI** by reusing per-chunk discretization
tables — within a chunk the work becomes a matmul against a `chunk ×
chunk` lower-triangular structured matrix.

## Roofline class
- Reference scan: **memory-bound.**
- SSD chunked: **balanced** — within-chunk LT semi-separable matmul
  maps onto the Tensor Unit and pushes AI into the GEMM regime.

## Engine assignment
- **Scan Engine** for the cross-chunk recurrence carry (Mamba S6
  reference and across-chunk state pass in SSD).
- **Tensor Unit** for the within-chunk semi-separable matmul (the
  SSD insight). This is the "≥80% shared code with attention"
  claim from PLAN.md §7.

## SRAM working set
Per channel: state `h` (`d_state · 4` = 64 bytes for d_state=16) +
discretization tables A_bar, B_bar for the chunk (`chunk · d_state ·
4` each = 1 KiB each for chunk=16). Tiny per channel; batched across
channels for tensor-unit throughput.

## DMA pattern
- x, delta: **sequential streaming**.
- A: **broadcast** (per-channel constant for the whole sequence).
- B, C: **sequential streaming** (input-dependent, change each step).
- h: **register-resident** within chunk; **carried** across chunks.

## Datapath dtype requirements
- All inputs: BF16 / FP16 typical.
- Discretization (`exp(delta · A)`): **FP32** for stability.
- State and reductions: **FP32 hard requirement**.

## Open silicon questions
- Right `chunk_size` for SSD: 16 / 32 / 64. Larger chunks → better LT
  matmul reuse, more state pressure.
- Whether to share the Scan Engine with the linear-attention recurrence
  (same shape: per-channel state + rank-1 update).
