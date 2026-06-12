# MoE Family — Kernel Contract

> PLAN.md §3 Phase 5 / §8 M6 — Mixture-of-Experts kernels.
> Operators: `moe_router`, `moe_top_k`, `moe_dispatch`, `moe_combine`.
> The expert-side compute itself is **`grouped_gemm`** (already in
> `kernels/gemm/`).

## Operators

### `moe_router(x, gate_weight)`
Project tokens to per-expert logits.

| Operand     | Shape                  | Notes |
|-------------|------------------------|-------|
| x           | (tokens, hidden)       | post-norm activations |
| gate_weight | (hidden, num_experts)  | router projection |
| Output      | (tokens, num_experts)  | dtype = `x.dtype` |

### `moe_top_k(logits, *, k, normalize=True)`
Pick top-k experts per token.

| Output      | Shape           | dtype |
|-------------|-----------------|-------|
| expert_ids  | (tokens, k)     | int32 |
| weights     | (tokens, k)     | float32 |

When `normalize=True` (the standard Mixtral / Switch convention) the per-
token weights are softmax over only the *selected* logits, so they sum
to 1.

### `moe_dispatch(x, expert_ids, *, num_experts)`
Permute tokens so all rows routed to the same expert are contiguous.

| Output           | Shape              | dtype |
|------------------|--------------------|-------|
| permuted_x       | (tokens * k, hidden)| `x.dtype` |
| expert_offsets   | (num_experts + 1,) | int32 — start row per expert |
| recovery         | (tokens * k,)      | int32 — flat slot in `(token, k)` |

`expert_offsets[e+1] - expert_offsets[e]` is the per-expert token count for
the grouped-GEMM call that follows.

### `moe_combine(expert_outputs, weights, recovery, *, n_tokens, hidden)`
Un-permute and weighted-sum back to per-token form.

| Output | Shape             | dtype |
|--------|-------------------|-------|
| out    | (n_tokens, hidden) | `expert_outputs.dtype` |

## Routing semantics

* **Stable** by expert id — ties between experts are broken by token order
  for reproducibility.
* **Lossless** under top-k routing — every token contributes to its top-k
  experts; no token-drop policy in M6 (capacity-based drop is a serving
  policy and lives in the runtime, not the kernel).

## Performance contract

Routing + dispatch are **memory-bound** (gather/scatter); the per-expert
matmul (grouped GEMM) is **compute-bound** for prefill, **memory-bound**
for decode. The dominant decode cost is the expert weight bandwidth times
the active-experts count — the W4A16 path shrinks this 4× and is the
production target for Mixtral-class decode.

## Cross-target equivalence

PLAN.md §8 M6 exit gate component: Mixtral decode correctness end-to-end.
The MoE kernels run through the cross-target equivalence harness
(`validation/equivalence.py`) so CPU / sim / RTL / FPGA / lhsil all
produce the same output bit-tight under tolerance.
