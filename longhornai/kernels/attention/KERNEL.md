# Attention Family — Kernel Contract

> PLAN.md §3 Phase 2 — attention kernels.
> Operators (M2): `sdpa`, `flash_attention_v1`.
> Operators (M3): `flash_attention_v2`, `multi_head_attention`, `mha`, `mqa`,
> `gqa`.
> Operators (M4): `paged_attention` + KV-cache extensions.

## Operators

### `sdpa(q, k, v, *, scale=None, causal=False)`
Math-form scaled dot-product attention; correctness anchor for the flash
variants.

| Operand | Shape         | Notes |
|---------|---------------|-------|
| q       | (..., S_q, D) | leading dims = batch + heads |
| k       | (..., S_kv, D) | broadcast against q for GQA pre-broadcast |
| v       | (..., S_kv, D) | |
| Output  | (..., S_q, D) | dtype = `np.result_type(q, k, v)` |

`scale` defaults to `1 / sqrt(D)`. `causal=True` masks `j > i`. Test inputs
must keep at least one un-masked column per Q row; an entirely masked row
produces `NaN` in SDPA and `0.0` in the streaming variants — undefined input.

### `flash_attention_v1(q, k, v, *, scale=None, causal=False, block_q=64, block_kv=64)`
IO-aware tiled forward via online softmax (Dao et al., 2022).

### `flash_attention_v2(q, k, v, *, scale=None, causal=False, block_q=64, block_kv=64)`
Flash-v1 plus the v2 algorithmic differences: causal blocks strictly above
the diagonal are skipped (saves work proportional to `S²/2` on long causal
sequences) and the work partitioning is rearranged for warp-level parallelism
on real hardware. Numerically identical to SDPA.

PLAN.md §8 M3 exit gate: **FA v2 numerical parity (latency + numerics)** vs
the SDPA reference on the benchmark causal/non-causal suite.

### `multi_head_attention(q, k, v, *, num_q_heads, num_kv_heads, head_dim, causal=False, scale=None, attn_impl="flash_v2")`
Generic per-head attention dispatch. `q` is `(B, S, H_q*D)`, `k` and `v` are
`(B, S_kv, H_kv*D)`. The kernel:

1. Reshapes to per-head form `(B, H, S, D)`.
2. Replicates K/V along the head axis when `H_kv < H_q` (GQA / MQA).
3. Calls the named attention impl (`sdpa` / `flash_v1` / `flash_v2`).
4. Concats heads back to `(B, S, H_q*D)`.

`num_q_heads` must be a multiple of `num_kv_heads`.

### `mha(q, k, v, *, num_heads, head_dim, **kw)`
Multi-head attention: `num_kv_heads == num_heads` (Llama-2, Qwen-1).

### `mqa(q, k, v, *, num_q_heads, head_dim, **kw)`
Multi-query attention: `num_kv_heads == 1` (Falcon, PaLM). KV-cache footprint
shrinks by `num_q_heads`.

### `gqa(q, k, v, *, num_q_heads, num_kv_heads, head_dim, **kw)`
Grouped-query attention: arbitrary KV-head grouping (Llama-3, Qwen-2,
Mistral). Configurable group factor.

## Supported dtypes

| dtype     | Inputs | Accumulation (CPU ref) | Output |
|-----------|:------:|:----------------------:|:------:|
| float64   |   ✓    | float64                | float64 |
| float32   |   ✓    | float64*               | float32 |
| float16   |   ✓    | float64*               | float16 |
| bfloat16  |   ✓    | float64*               | bfloat16 (emulated) |

\* Hardware-representative backends use FP32 accumulation per the Phase-1
contract; the CPU reference is the correctness anchor.

## Numerical contract

Every variant validates against `kernels/attention/reference.py` (which
collapses to the SDPA math). Tile sizes (`block_q`, `block_kv`) and the
`attn_impl` selector are perf knobs only — they must not change output beyond
per-dtype tolerance (block-size invariance is a tested contract).

## Performance contract

PLAN.md §9.1 silicon target:

| Metric                                         | Target |
|------------------------------------------------|-------:|
| FA v2 vs FlashAttention v2 prefill throughput  | ≥ 95%  |
| Decode latency vs reference                    | ≤ +5%  |

Flash-v1 is the M2 prefill target; v2 is the M3+ prefill target. SDPA is the
small-S fallback only.

## Cross-target equivalence

PLAN.md §8 M3 exit gate also requires **RTL ≡ CPU on attention**. The
cross-target equivalence harness in `validation/equivalence.py` runs every
attention op under each registered backend and asserts agreement within
tolerance.

## Tuning

`kernels/attention/tuning.py` declares:

* `FLASH_V1_TUNING_SPACE` — `(block_q, block_kv, stages)`
* `FLASH_V2_TUNING_SPACE` — `(block_q, block_kv, stages, warp_split)`
* `MHA_TUNING_SPACE` — `(attn_impl, fuse_qkv_proj, fuse_o_proj)`

The autotuner is engaged per `(S_q, S_kv, D, dtype, n_q_heads, n_kv_heads)`.
