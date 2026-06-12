# Paged Attention Family — Kernel Contract

> PLAN.md §3 Phase 2 / M4 — paged attention.
> Operators: `paged_kv_alloc`, `paged_kv_append`, `paged_attention`.
> Foundation for high-throughput continuous batching (PLAN.md §3 Phase 3 /
> §10.3 Longhorn Serving Runtime).

## Layout

The KV cache is a pool of fixed-size **blocks**:

| Tensor   | Shape                                          | dtype         |
|----------|------------------------------------------------|---------------|
| cache_k  | (num_blocks, num_kv_heads, block_size, head_dim) | fp16/bf16/fp32 |
| cache_v  | (num_blocks, num_kv_heads, block_size, head_dim) | fp16/bf16/fp32 |

Each request has a **block table** that maps logical token positions to
physical blocks:

| Tensor       | Shape                          | dtype |
|--------------|--------------------------------|-------|
| block_table  | (B, max_blocks_per_seq)        | int32 |
| seq_lens     | (B,)                           | int32 |

Logical position `p` of request `b` lives at:

    physical = block_table[b, p // block_size]
    within   = p % block_size
    cache_k[physical, :, within, :] / cache_v[physical, :, within, :]

This indirection is what eliminates fragmentation: requests of any length
share one global pool of blocks, and the scheduler hands out blocks on
demand (`runtime/scheduler.py`).

## Operators

### `paged_kv_alloc(*, num_blocks, block_size, num_kv_heads, head_dim, dtype=fp16)`
Pure helper — allocates the pool. The scheduler computes `num_blocks` from
the available device memory and the block size.

### `paged_kv_append(cache_k, cache_v, k_new, v_new, *, block_table, seq_lens, block_size)`
Writes a uniform `S_new` new K/V tokens per request into the pool, starting
at logical position `seq_lens[b]`. Mutates the pool in place.

### `paged_attention(q, cache_k, cache_v, *, block_table, seq_lens, block_size, num_q_heads, num_kv_heads, head_dim, scale=None, causal=True)`
Variable-length attention reading paged KV. Same `S_q` per request in one
call (the scheduler segregates prefill vs decode batches). The kernel:

1. For each request `b`, gather K/V at logical positions `[0, seq_lens[b])`
   via the block table.
2. Replicate K/V along the head axis when `num_kv_heads < num_q_heads`
   (GQA / MQA).
3. Compute causal SDPA. Queries occupy positions `[seq_lens[b] - S_q,
   seq_lens[b])`; KV occupies `[0, seq_lens[b])`. The causal mask compares
   query position to KV position.

## Supported dtypes

| dtype     | Inputs | Accumulation (CPU ref) | Output |
|-----------|:------:|:----------------------:|:------:|
| float64   |   ✓    | float64                | float64 |
| float32   |   ✓    | float64                | float32 |
| float16   |   ✓    | float64                | float16 |
| bfloat16  |   ✓    | float64                | bfloat16 (emulated) |

## Numerical contract

`paged_attention` reduces to `sdpa` per request after the gather step.
Validated against `paged_attention/reference.py`, which performs that
reduction in float64.

`paged_kv_append` is structural — bit-equal to the reference (same
positions, same values).

PLAN.md §8 M4 exit gate component: paged attention parity vs SDPA on the
benchmark prefill/decode shapes.

## Performance contract

* **Prefill** (`S_q = prompt_len`) is compute-bound; the same roofline
  target as Flash-v2 applies (≥ 80% of compute roof).
* **Decode** (`S_q = 1`) is bandwidth-bound; weight-and-KV reads dominate.
  Roofline target ≥ 85% of bandwidth roof.

Block size is part of the tuning space — typically 16 (lhsil) or 64
(large-context modes). Smaller blocks reduce internal fragmentation; larger
blocks reduce block-table overhead and improve DRAM access patterns.

## Cross-target equivalence

PLAN.md §8 M4 exit gate: **E2E Llama decode under continuous batching on
FPGA**. Every paged op runs through the cross-target equivalence harness
under each registered backend (CPU, sim, RTL, FPGA).

## Tuning

`kernels/paged_attention/tuning.py` declares spaces over `block_size`,
`split_kv` (whether to do a v2-style split-KV reduction across blocks), and
`warps_per_block`.
