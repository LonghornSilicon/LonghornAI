# Positional encodings (`src/kernels/positional.{hpp,cpp}`)

## What it computes
- `ntk_scaled_theta(base, orig, ext, d)` — closed-form theta-base
  rescale for context-length extension.
- `yarn_inv_freq` — frequency-band-wise blended inverse frequencies
  (linear within trained context, extrapolation past it, ramp between).
- `alibi_slopes` — canonical geometric-series slopes per head.
- `alibi_bias` — materialised additive bias `[n_heads, seq_q, seq_k]`.

## Reference path
Single-pass implementations; no separate optimised path needed.

## Memory traffic
- NTK / YaRN: trivially small (writes `d/2` floats).
- ALiBi bias: `n_heads · seq_q · seq_k` floats — significant at long
  context (seq_q · seq_k ≈ 1M for seq=1024).

## Arithmetic intensity
- NTK / YaRN: irrelevant (one-shot scalar work).
- ALiBi bias: 1 FLOP/elem; **memory-bound** to write the bias buffer.

## Roofline class
- NTK / YaRN: cold-path setup, run once per-context.
- ALiBi bias: **memory-bound**, motivates *not* materialising it —
  better to compute the bias on-the-fly inside the attention engine
  (one FMA per (head, q, k) pair, no separate buffer).

## Engine assignment
- NTK / YaRN: **Host-side / Scheduler** — runs once at request entry.
- ALiBi bias: ideally **Attention Engine** epilogue with the slope
  table preloaded. The materialised buffer in this codebase exists for
  test parity with software stacks that hold the bias in memory.

## SRAM working set
- ALiBi slopes: `n_heads · 4` bytes. Negligible.
- Materialised bias (test path): up to `n_heads · seq_q · seq_k · 4`
  bytes. Avoid.

## DMA pattern
- NTK / YaRN: irrelevant.
- ALiBi (materialised): broadcast load to attention.
- ALiBi (fused): zero DMA — the slope is a register constant.

## Datapath dtype requirements
- Slopes: FP32.
- Bias add: FP32 (gets summed with attention scores in the same lane).

## Open silicon questions
- Whether per-head theta tables (for YaRN) live in a small per-head
  ROM block or in shared SRAM with the cos/sin table.
