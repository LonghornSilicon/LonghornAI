# Collectives Family — Kernel Contract

> PLAN.md §3 Phase 6 / §8 M6 — multi-GPU communication primitives.
> Operators: `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`.

## Operands

All four kernels operate on a **list of per-rank shards** — one
`np.ndarray` per virtual rank. Every shard shares dtype; shape rules vary
by operator.

## Operators

### `all_reduce(shards, *, op="sum")`
Reduce across ranks; return the same reduced tensor on every rank.

| Operand | Shape per shard | Notes |
|---------|-----------------|-------|
| shards  | identical       | every rank's contribution |
| Output  | identical to shards | shape unchanged |

`op` ∈ {"sum", "max", "mean"}.

### `all_gather(shards, *, axis=0)`
Concatenate shards along `axis`; every rank gets the full result.

| Operand | Shape per shard | Notes |
|---------|-----------------|-------|
| shards  | match on every axis except `axis` | |
| Output  | (..., sum_over_ranks(axis_dim), ...) | |

### `reduce_scatter(shards, *, op="sum", axis=0)`
All-reduce, then partition along `axis` so rank `r` holds its 1/R slice.
The reduced full-axis dim must be divisible by the rank count.

### `all_to_all(shards, *, axis=0)`
Each shard's `axis` dim is split into N chunks (N = num_ranks); chunk `j`
of rank `i`'s input becomes chunk `i` of rank `j`'s output. Used for
expert-parallel token routing (PLAN.md §3 Phase 5).

## Contract — multi-backend behavior

The CPU backend simulates a multi-rank world by computing the collective
math directly on the shard list. Real lhsil silicon swaps in NCCL /
RCCL / SHIM equivalents; the per-rank *output shards* must match.

The cross-target equivalence harness validates this — every collective
op runs through `validation/equivalence.py` so the FPGA shim is checked
to match CPU on every collective call.

## Performance contract

PLAN.md §9.2: **scaling efficiency** is the binding metric for collectives.
Production silicon must hit a published efficiency target on tensor-
parallel and expert-parallel scaling across the interconnect. Today on
CPU the cost is just the Python-level numpy work; the contract becomes
load-bearing on real silicon.

## Tuning

`kernels/collectives/tuning.py` declares spaces over `algorithm` (ring vs
tree vs hierarchical) and `chunk_size_kb`. The CPU shim ignores them;
sim/RTL/FPGA/lhsil consume them through dispatch.
