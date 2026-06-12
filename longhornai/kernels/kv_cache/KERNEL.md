# KV-Cache Family — Kernel Contract

> PLAN.md §3 Phase 2 / M3 — KV-cache kernels.
> Operators (M3): `kv_cache_alloc`, `kv_cache_append`, `kv_cache_gather`,
> `kv_cache_alloc_int8`, `kv_cache_quantize_append`,
> `kv_cache_dequantize_gather`.
> Operators (M4): paged variants (block table indirection over a paged KV
> pool — PLAN.md §3 Phase 3).

## Layout

Dense KV cache:

| Tensor       | Shape                      | dtype         |
|--------------|----------------------------|---------------|
| cache_k      | (B, H_kv, S_max, D)        | fp16/bf16/fp32 |
| cache_v      | (B, H_kv, S_max, D)        | fp16/bf16/fp32 |

INT8 KV cache:

| Tensor       | Shape                      | dtype         |
|--------------|----------------------------|---------------|
| cache_k_int8 | (B, H_kv, S_max, D)        | int8          |
| scale_k      | (B, H_kv, S_max)           | float32       |
| cache_v_int8 | (B, H_kv, S_max, D)        | int8          |
| scale_v      | (B, H_kv, S_max)           | float32       |

Quantization is **per-token symmetric** — one scale per `(B, H_kv, S)` slot.
This keeps the dynamic range tight for streaming token writes while costing
only `4 × B × H_kv × S_max` bytes of metadata vs the cache itself.

## Operators

### `kv_cache_alloc(*, batch, num_kv_heads, max_seq_len, head_dim, dtype=fp16)`
Pure-Python helper — no dispatch. Allocates zero-initialized
`(cache_k, cache_v)` tensors.

### `kv_cache_append(cache_k, cache_v, k_new, v_new, *, position)`
Writes `k_new`, `v_new` into the cache at sequence offset `position`. Returns
`(cache_k, cache_v)` — backends may mutate in place, the reference returns
copies so the differential harness compares values, not aliasing.

### `kv_cache_gather(cache_k, cache_v, *, length)`
Returns the valid `[:length]` prefix. The CPU backend returns views (zero-copy);
real silicon kernels will return device-resident slices.

### `kv_cache_alloc_int8(*, batch, num_kv_heads, max_seq_len, head_dim)`
Allocates the four-tensor INT8 layout described above.

### `kv_cache_quantize_append(cache_k_int8, scale_k, cache_v_int8, scale_v, k_new, v_new, *, position)`
Per-token symmetric INT8 quantize → write at `position`. Updates both the
quantized cache and the matching scale tensors.

### `kv_cache_dequantize_gather(cache_k_int8, scale_k, cache_v_int8, scale_v, *, length, dtype=fp16)`
Read `[:length]` and dequantize to `dtype`. Returns dense `(B, H_kv, length, D)`
tensors ready for the attention kernel.

## Numerical contract

Append/gather are structural — bit-equal to the references for any supported
fp dtype. The INT8 round-trip (quantize → dequantize) carries the symmetric
INT8 representation error, bounded by ≤ 1 / 127 relative for in-range values
and validated against `kernels/quantization/primitives.py`.

For the M3 exit gate **RTL ≡ CPU on attention**, every KV-cache op runs
under the cross-target equivalence harness alongside SDPA / Flash-v1 / Flash-
v2 / multi_head_attention.

## Performance contract

KV cache ops are bandwidth-bound and dominate decode-time HBM traffic per
PLAN.md §3 Phase 1's "decode is bandwidth-bound" classification. The roofline
target is ≥ 85% of bandwidth roof on the production silicon target.

The INT8 cache halves storage (and bandwidth) at the dequant cost; the
bandwidth-bound-ness of dequant means it's a near-net-win on lhsil if the
dequant fuses into the attention gather pass — a tuning knob.

## Tuning

`kernels/kv_cache/tuning.py` declares spaces for vector width, layout
(`bhsd` vs `bshd`), and fusion of the dequant step into the gather pass. The
CPU backend ignores them; sim/RTL/FPGA/lhsil consume them through dispatch.
