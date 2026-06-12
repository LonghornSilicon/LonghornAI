# Fused Kernel Family — Kernel Contract

> PLAN.md §3 Phase 6 / §8 M7 — operator fusion.
> Operators: `rmsnorm_qkv`, `attention_output_proj`, `gated_mlp`.

## Why

Decode is bandwidth-bound. Each sub-op of an attention block (RMSNorm,
QKV projections, SDPA, output projection) reads its inputs from HBM and
writes its outputs back. Fusing the chain so each operand makes **one**
HBM round-trip cuts decode-time bandwidth ~3-4× — the single highest-
leverage optimization on real silicon (PLAN.md §3 Phase 6).

The fused kernel is *numerically equivalent* to its un-fused chain. The
fusion is a memory-traffic optimization, not a math change.

## Operators

### `rmsnorm_qkv(x, *, norm_weight, q_weight, k_weight, v_weight, eps=1e-5)`
Pre-norm + Q/K/V projections, fused. The standard prologue of every
attention block.

| Operand     | Shape                | Notes |
|-------------|----------------------|-------|
| x           | (tokens, hidden)     | input |
| norm_weight | (hidden,)            | RMSNorm gain |
| q_weight    | (hidden, n_q*D)      | column-parallel-friendly |
| k_weight    | (hidden, n_kv*D)     | |
| v_weight    | (hidden, n_kv*D)     | |
| Output      | tuple `(q, k, v)` of projected dims | dtype = `x.dtype` |

### `attention_output_proj(q, k, v, *, o_weight, causal=False, scale=None)`
SDPA + output projection, fused. The standard attention-block epilogue.

| Operand   | Shape                     | Notes |
|-----------|---------------------------|-------|
| q, k, v   | (B, n_heads, S, D)        | per-head |
| o_weight  | (n_heads * D, hidden)     | row-parallel-friendly |
| Output    | (B, S, hidden)            | dtype = `q.dtype` |

### `gated_mlp(x, *, gate_weight, up_weight, down_weight, activation="silu")`
SwiGLU / GeGLU MLP, fully fused. ``activation`` ∈ {"silu", "gelu"} matches
the `LlamaConfig.mlp_activation` field.

| Operand      | Shape                       |
|--------------|-----------------------------|
| x            | (tokens, hidden)            |
| gate_weight  | (hidden, intermediate)      |
| up_weight    | (hidden, intermediate)      |
| down_weight  | (intermediate, hidden)      |
| Output       | (tokens, hidden)            |

## Numerical contract

Each fused op must produce output bit-tight with its un-fused reference
under the per-dtype tolerance policy. The differential harness validates
this at every dtype.

## Performance contract

PLAN.md §9.1 — fusion reduces the bandwidth roof denominator:

* `rmsnorm_qkv`         — 3 matmul reads collapse to 1; the norm is a
  GEMM prologue.
* `attention_output_proj` — 1 attention round-trip + 1 matmul collapse to
  1; o_proj fuses into the attention epilogue.
* `gated_mlp`           — 3 matmuls + activation collapse to a 2-stage
  pipeline whose intermediate stays in SRAM.

## Cross-target equivalence

PLAN.md §8 M7 exit gate: cross-target equivalence holds. Fused ops run
through `validation/equivalence.py` and must agree with their un-fused
references on every backend.
