# Longhorn — Candidate Inference Accelerator Microarchitecture

**Phase 9 synthesis. Derived from the Phase 8 kernel inventory + engine
map; detailed enough for an RTL team to begin Stage B (FPGA prototype)
implementation.**

This document describes the Longhorn accelerator at the level of: block
diagrams, per-block compute / SRAM / interface specs, on-chip
interconnect topology, memory hierarchy, instruction format, execution
model, and the relationship between edge and server profiles.

Every architectural decision in this document traces back to a kernel
in the LonghornAI repository and a justification in
`docs/silicon/kernels/<name>.md`. The companion
[`isa.md`](isa.md) gives the instruction-level view; [`roadmap.md`](roadmap.md)
gives the FPGA / ASIC delivery plan; [`perf_model.md`](perf_model.md)
explains the analytical performance model that validates the design
against the Phase 8 dossiers.

---

## 1. Design philosophy

Three principles, in priority order:

1. **The workload dictates the silicon.** Phase 8 produced a kernel
   inventory dominated by memory-bound operations at realistic
   inference shapes. Compute headroom is plentiful; HBM bandwidth is
   scarce. **Longhorn spends area on bandwidth before MAC units.**

2. **Specialise where the workload concentrates.** Two-thirds of
   inference's wall-clock time is in three operations: attention
   (decode), MoE expert MLP (memory-bound on weight reads), and the
   KV cache read/write path. Each gets a dedicated block. Generic
   tensor work shares a single Tensor Unit.

3. **Static scheduling, software-orchestrated memory.** The compiler
   knows the kernel inventory; the silicon doesn't need to discover
   instruction-level parallelism at runtime. No out-of-order issue.
   No coherent caches — every DMA is software-driven, every SRAM
   tier is software-managed.

These principles dictate the rest of the document.

---

## 2. Top-level block diagram

```
        ┌─────────────────────────────────────────────────────────────┐
        │                    Longhorn Accelerator                     │
        │                                                             │
        │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
        │   │ TENSOR    │  │ TENSOR    │  │ TENSOR    │  │  ATTN   │  │
        │   │ TILE 0    │  │ TILE 1    │  │ TILE N-1  │  │ ENGINE  │  │
        │   │           │  │           │  │           │  │         │  │
        │   │ ┌───────┐ │  │ ┌───────┐ │  │ ┌───────┐ │  │ shared  │  │
        │   │ │TU+VU+ │ │  │ │TU+VU+ │ │  │ │TU+VU+ │ │  │ across  │  │
        │   │ │NORM+  │ │  │ │NORM+  │ │  │ │NORM+  │ │  │ tiles   │  │
        │   │ │REDUCT │ │  │ │REDUCT │ │  │ │REDUCT │ │  │         │  │
        │   │ └───────┘ │  │ └───────┘ │  │ └───────┘ │  └────┬────┘  │
        │   │ ┌───────┐ │  │ ┌───────┐ │  │ ┌───────┐ │       │       │
        │   │ │ L1    │ │  │ │ L1    │ │  │ │ L1    │ │       │       │
        │   │ │ SRAM  │ │  │ │ SRAM  │ │  │ │ SRAM  │ │       │       │
        │   │ │ 256KB │ │  │ │ 256KB │ │  │ │ 256KB │ │       │       │
        │   │ └───────┘ │  │ └───────┘ │  │ └───────┘ │       │       │
        │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │       │
        │         │              │              │             │       │
        │   ╔═════╧══════════════╧══════════════╧═════════════╧════╗  │
        │   ║                  NoC (mesh, multicast)               ║  │
        │   ╚═══┬═════════════════════════┬═════════════════════┬══╝  │
        │       │                         │                     │     │
        │   ┌───┴────────┐    ┌───────────┴────────┐    ┌──────┴───┐  │
        │   │ KV CACHE   │    │ PERMUTATION ENGINE │    │ SCAN     │  │
        │   │ CONTROLLER │    │  (MoE / Transpose) │    │ ENGINE   │  │
        │   │ blocktab + │    │                    │    │ (SSM /   │  │
        │   │ dequant +  │    │                    │    │ RWKV /   │  │
        │   │ prefetch   │    │                    │    │ LinAttn) │  │
        │   └─────┬──────┘    └─────────┬──────────┘    └────┬─────┘  │
        │         │                     │                    │        │
        │   ┌─────┴─────────────────────┴────────────────────┴─────┐  │
        │   │              L2 SRAM (8 MiB shared)                  │  │
        │   └─────────────────────────┬────────────────────────────┘  │
        │                             │                               │
        │   ┌─────────────────────────┴────────────────────────────┐  │
        │   │             DMA ENGINES (8 channels)                 │  │
        │   └─────────────────────────┬────────────────────────────┘  │
        │                             │                               │
        │   ┌─────────────────────────┴────────────────────────────┐  │
        │   │           HBM / LPDDR Memory Controller              │  │
        │   └─────────────────────────┬────────────────────────────┘  │
        └─────────────────────────────┼───────────────────────────────┘
                                      │
                                  external
                              HBM3e × 4 stacks (server profile)
                                or
                              LPDDR5X (edge profile)
```

The chip is a small mesh of **N Tensor Tiles** plus a set of dedicated
shared engines (Attention, KV Controller, Scan, Permutation, DMA). A
**Hardware Scheduler** (not shown) is the entry point that issues
coarse-grained instructions to all blocks.

---

## 3. The Tensor Tile

The Tensor Tile is the unit of replication. It contains everything
needed to run a generic linear layer: dense GEMM, batched/grouped
GEMM, normalisation, softmax (when not a part of an attention
fusion), reductions, activations, and quantization rescale.

### 3.1 Tensor Tile internals

```
       ┌────────────────────────────────────────────────────┐
       │                  Tensor Tile                       │
       │                                                    │
       │     ┌──────────────────────────────────────────┐   │
       │     │              Tensor Unit                 │   │
       │     │  systolic array, 64×64 MACs              │   │
       │     │  inputs: BF16/FP16/FP8/INT8/INT4         │   │
       │     │  accumulator: FP32 / INT32               │   │
       │     │  + INPUT FEEDERS with on-the-fly dequant │   │
       │     └────────────────────────────────────────┬─┘   │
       │                                              │     │
       │     ┌──────────────────────────────────────────┐   │
       │     │              Vector Unit                 │   │
       │     │  64-lane SIMD, FP32 / BF16               │   │
       │     │  exp / log / rsqrt / erf / tanh pipes    │   │
       │     │  PRNG (splitmix64) + multinomial sampler │   │
       │     └──────────────────────────────────────────┘   │
       │                                                    │
       │     ┌────────────────┐  ┌─────────────────────┐    │
       │     │ Norm Engine    │  │  Reduction Engine   │    │
       │     │ Welford 1-pass │  │  sum/max/mean/topk  │    │
       │     │ + rsqrt + γβ   │  │  tree reducer       │    │
       │     └────────────────┘  └─────────────────────┘    │
       │                                                    │
       │     ┌────────────────────────────────────────────┐ │
       │     │              Softmax Engine                │ │
       │     │  online: max-subtract → exp → sum → recip  │ │
       │     │  shared with attention engine via NoC      │ │
       │     └────────────────────────────────────────────┘ │
       │                                                    │
       │     ┌────────────────────────────────────────────┐ │
       │     │           L1 SRAM scratchpad (256 KiB)     │ │
       │     │  banked, 8 banks × 32 KiB                  │ │
       │     └────────────────────────────────────────────┘ │
       │                                                    │
       │     ┌────────────────────────────────────────────┐ │
       │     │              L0 / register file            │ │
       │     │  64-entry × 256-byte vector regs           │ │
       │     └────────────────────────────────────────────┘ │
       └────────────────────────────────────────────────────┘
```

### 3.2 Tensor Unit

A **64 × 64 systolic array** of MAC cells. Per cell:

- One BF16/FP16/FP8/INT8/INT4 multiplier (input-width selectable per
  instruction).
- One FP32 / INT32 accumulator (selectable; INT32 for W8A8, FP32
  otherwise — both required, not optional, per `gemm_quant.md`).
- Local skew register for systolic dataflow.

Per-cycle throughput: 4096 fused MAC operations.

**Input feeders** sit at the array's edges:

- **A feeder**: streams activation rows; supports per-row scales for
  W8A8 dequant and per-tensor / per-row scales for FP8 dequant.
- **B feeder**: streams weight panels; supports
  - per-column scale (W8A8),
  - per-tensor scale (FP8),
  - per-`(group, col)` scale + INT4 unpack (W4A16).
  
  This is the "dequant on the input feeders" pattern from
  `gemm_quant.md` — the MAC array sees a uniform compute dtype.

**Epilogue / Rescale Unit** at the array's output:

- Multiplies the FP32/INT32 accumulator by `α · A_scale · B_scale`
  to produce FP32 output.
- Optionally applies activation (ReLU / GELU / SiLU / Tanh) before
  writeback — the "fused MatMul + activation" path from
  `activation.md`.
- Optionally applies SwiGLU / GeGLU gate-mix when paired with a
  preceding `up` projection result held in scratchpad — fuses the
  Llama MLP block to a single round-trip through L1 SRAM.

### 3.3 Vector Unit

A **64-lane SIMD path** for everything that isn't a matmul:

- Pointwise FP32/BF16 ops (add, mul, abs, neg, clamp, rsqrt).
- Transcendental pipes: `exp`, `log`, `tanh`, `erf` via polynomial
  approximation (Taylor + range reduction). Shared by the softmax
  engine, normalization engine, and activation epilogues.
- A **PRNG lane** running splitmix64 (matches the kernel-side
  `sampling.cpp` PRNG byte-for-byte) + a multinomial sampler for the
  full `sample` pipeline from `sampling.md`.

The Vector Unit is intentionally narrow (64 lanes) compared to the
Tensor Unit (4096 MACs). Vector kernels are memory-bound; widening
the vector path past the SRAM bank count just buys bubbles.

### 3.4 Normalization Engine

A streaming reduction path with the Welford one-pass variance built
in (per `normalization.md`):

```
input row → Σx, Σx²  (one pass, FP32)
         → μ, σ²
         → rsqrt(σ² + ε)
         → broadcast (γ, β)
         → output row
```

`(γ, β)` are preloaded once per layer into a small constant register
file. The engine writes back to L1 SRAM directly, ready for the next
GEMM.

**RMSNorm**: same engine, mask out the mean-subtract path
(`mean_active = false` instruction bit).

**Fused RMSNorm + RoPE epilogue** (Phase 8 open question): the engine
optionally consumes a per-position cos/sin tile and applies RoPE to
the output before writeback. Eliminates one full L1 round-trip on the
residual stream — the highest-leverage in-tile fusion.

### 3.5 Reduction Engine

Tree reducer with three modes:

- `sum / max / mean` (FP32 accumulators).
- `argmax` (strict-greater compare with INT32 lane index, lowest-index
  tie-break — matches `reduction.cpp`).
- `topk` (length-`k` heap; `k ≤ 64` fast path, larger `k` walks
  through L1 SRAM).

The argmax mode also doubles as the greedy-sample fast path.

### 3.6 Softmax Engine

Streaming online softmax: `max-subtract → exp → sum → reciprocal`
fused into one pipeline, with `(m, ℓ)` registers held across
invocations so the engine can be driven by either:

- Standalone softmax (e.g., LM-head sample preparation).
- Attention Engine (the same engine instance is multiplexed during
  attention's online-softmax tile loop — see §4.2).

### 3.7 L1 SRAM

**256 KiB per Tensor Tile**, organised as 8 banks of 32 KiB. Sized
from `gemm.md`'s panel budget (32 KiB) plus headroom for the
attention online-softmax state (16 KiB) plus per-tile scratch.

Banking lets the Tensor Unit, Vector Unit, and engines read/write
concurrently without conflicts when the compiler schedules
non-overlapping tile assignments. A **bank conflict counter** is
exposed to the perf model for compiler tuning.

### 3.8 L0 / register file

64 vector registers, 256 bytes each (= 64 × FP32 lanes = one MAC-cell-
wide vector). Used for staging the very-hot inputs to the Tensor Unit
and for activation epilogue staging.

---

## 4. The Attention Engine

A dedicated block, **shared across all Tensor Tiles via the NoC**.
Sized once per chip rather than per tile because:

- Its hot path (`paged_attention` decode) is bandwidth-bound on the
  KV side, not compute-bound; replicating it doesn't help.
- Sharing lets multiple Tensor Tiles feed Q tiles into the same
  attention pipe and consume O tiles back, which improves utilisation
  during prefill.

### 4.1 Block diagram

```
                ┌───────────────────────────────────────────────┐
                │                Attention Engine               │
                │                                               │
   Q tiles ───► │  Q LOADER                                     │
                │                                               │
   from KV   ─► │  K STREAMER ──┐                               │
   Controller   │               ▼                               │
                │            QKᵀ MATMUL  ◄── per-head scale     │
                │               │                               │
                │               ▼                               │
                │            BIAS ADD     ◄── ALiBi slope /     │
                │               │             tree mask         │
                │               ▼                               │
                │       ONLINE SOFTMAX                          │
                │       (m, ℓ, O state)  ── shares exp pipe     │
                │               │           with Softmax Engine │
                │               ▼                               │
                │           PV MATMUL                           │
                │               │                               │
   from KV   ─► │  V STREAMER ──┘                               │
   Controller   │                                               │
                │               ▼                               │
   to host   ◄─ │   O WRITEBACK                                 │
                │                                               │
                │   FP32 accumulators throughout (m, ℓ, O)      │
                └───────────────────────────────────────────────┘
```

Per `attention.md`'s engine spec:

- Q tiles enter from the upstream Tensor Tile's L1 SRAM via NoC.
- K and V tiles enter from the **KV Cache Controller** (§5).
- Score matmul, bias add, online softmax (`m`, `ℓ`, `O`), and PV
  matmul live in one pipeline — no intermediate writeback to L1.
- The `(m, ℓ, O)` state is FP32 register-resident throughout.

### 4.2 Sizing

- **QKᵀ matmul**: 32 × 64 systolic sub-array (smaller than the
  Tensor Unit because per-Q-tile reuse is lower).
- **PV matmul**: 32 × 64 sub-array.
- **Online-softmax state**: 16 KiB FP32 register file (for `Mq` query
  rows × `(d + 2)` state per row).
- **Tile size**: `Mq = 32`, `block_k = 64` (matches the
  `flash_attention` reference's tested shapes).

### 4.3 Modes

A 3-bit `mode` field on the attention instruction selects:

| Mode | Behaviour | Backed by |
|---|---|---|
| `SDPA` | Full materialised path (small Lk only) | `sdpa` |
| `FLASH_PREFILL` | Q-outer, online softmax, no score matrix | `flash_attention` (Q-outer) |
| `FLASH_DECODE` | Split-K + cross-split merge | `flash_decode_attention` |
| `CROSS` | Q from one stream, K/V from another | `cross_attention` |
| `TREE` | Per-(q, k) additive bias from L2 | `build_tree_attention_bias` |

Each is one of the kernels in Phase 7's bench suite; the engine maps
to the mode via opcode + bit flags.

---

## 5. The KV Cache Controller

The single most consequential block in the chip. Per `kvcache.md`,
decode attention is **memory-bound, dominated by the KV read** — and
the controller is what determines whether that bandwidth is
realised.

### 5.1 Block diagram

```
              ┌─────────────────────────────────────────────────┐
              │              KV Cache Controller                │
              │                                                 │
  attention   │  BLOCK TABLE WALKER                             │
  request ──► │  - one cycle per logical block lookup           │
              │  - block_size = 16 (configurable per layer)     │
              │                                                 │
              │  PREFETCHER                                     │
              │  - 4-deep request queue                         │
              │  - 2 outstanding HBM reads per request          │
              │  - speculative next-block prefetch              │
              │                                                 │
              │  DEQUANT UNIT                                   │
              │  - INT4 / INT8 / FP8 → FP16 / BF16              │
              │  - per-row scale broadcast                      │
              │  - matches `paged_attention_q8` numerics        │
              │                                                 │
              │  APPEND PATH                                    │
              │  - quantize on write                            │
              │  - matches `paged_kv_append` / _q8              │
              │                                                 │
              │  EVICTION / MIGRATION (DMA-driven)              │
              │  - block-grain DMA descriptors                  │
              │  - free-list maintained alongside CacheManager  │
              └────────────────────┬────────────────────────────┘
                                   │
                              HBM (server) / LPDDR (edge)
```

### 5.2 Pool layout in DRAM

```
[num_blocks, n_kv_heads, block_size, head_dim] (storage dtype)
[num_blocks, n_kv_heads, block_size]            (FP32 scale, q8 mode)
```

Matches `PagedCacheLayout` in `kvcache.hpp`. The block-table is a
host-resident `int32` array; the controller pulls it via a small
on-chip mirror cache.

### 5.3 Scaling

- Server profile: 4 prefetch streams × 2 outstanding HBM reads each
  = **8 in-flight reads per attention head** to keep HBM busy.
- Edge profile: 1 prefetch stream, 1 outstanding read — LPDDR
  bandwidth is the limit and overlapping helps less.

### 5.4 Coherence with Attention Engine

A **single-cycle handoff queue** of (K_tile, V_tile, mask_slice)
between the controller's dequant output and the attention engine's
streamer input. No round-trip through L1 SRAM. The handoff queue is
4 entries deep; back-pressure pauses the prefetcher when attention
falls behind.

---

## 6. The Scan Engine

Handles the recurrent kernels that don't fit attention's shape:
Mamba S6 selective scan, Mamba-2 SSD cross-chunk carry, RWKV WKV,
and linear attention. Per the dossiers (`ssm.md`, `rwkv.md`,
`linear_attn.md`), these share a single computational shape:
**per-channel state + rank-1 (or rank-N) update + readout**.

### 6.1 Microcode modes

The Scan Engine is one block with three numerical specialisations
selected by a 2-bit `mode` field:

| Mode | Update rule | Used by |
|---|---|---|
| `SSM` | `h ← A_bar · h + B_bar · x` | Mamba S6 sequential, SSD cross-chunk |
| `WKV_LOG` | log-space (a, b, p) recurrence | RWKV v5/v6 |
| `LINATTN` | `S ← S + k ⊗ v`, `z ← z + k` | linear attention, RetNet |

Within-chunk SSD (Mamba-2's matmul path) routes to the **Tensor
Unit** instead — the dossier's "SSD shares ≥80% of attention's
tensor-unit code" claim materialises here as a microcode dispatch:
the cross-chunk carry runs on the Scan Engine; the within-chunk
matmul runs on the Tensor Unit.

### 6.2 Sizing

- Per-channel state: up to 256 bytes (covers Mamba `d_state=16`,
  RWKV's tiny `(a, b, p)`, and linear attention's `S` for
  `d_feat=d_v=8` per state slot).
- 64 parallel channels per Scan Engine instance — picked to match
  the Vector Unit's lane count so a scan can be dispatched alongside
  vector work without contention.
- **Log-space lanes are FP32-only** (per `rwkv.md`'s hard
  requirement).

### 6.3 Number of instances

One per Tensor Tile. The scan workload is per-channel-parallel; one
engine per tile gives `N` channels of throughput where `N` is the
number of tiles.

---

## 7. The Permutation Engine

Handles MoE dispatch / combine, transpose / permute, and the
gather / scatter kernels from `transform.md` and `moe.md`. One block
per chip, shared across tiles.

### 7.1 Modes

| Mode | Operation | Backed by |
|---|---|---|
| `COUNTING_SORT_SCATTER` | token-major → expert-major | `moe_dispatch` |
| `WEIGHTED_GATHER` | expert-major → token-major | `moe_combine` |
| `STRIDED_TRANSPOSE` | 2D / N-D transpose, permute | `transpose2d`, `permute` |
| `ROW_GATHER` / `ROW_SCATTER_ADD` | embedding-style indirect | `gather_rows`, `scatter_add_rows` |

### 7.2 Sizing

- Permutation throughput: 64 × 256-bit lanes = 2 KiB/cycle. Sized to
  saturate the L2 ↔ Tile NoC.
- Internal scratch: 64 KiB for the active permutation buffer.
- Atomic-add path for `scatter_add_rows`: a small CAM (16 entries)
  to coalesce duplicate destination rows in flight.

---

## 8. DMA engines

**8 channels**, descriptor-driven. Each supports:

- **Strided streaming** (1D contig, 2D stride).
- **Scatter-gather** (descriptor list of `(src, dst, len)`).
- **Transposed-tile** (read MR × NR, write NR × MR).

The 8-channel count comes from the Phase 8 fan-in of memory-bound
kernels: at any moment the chip is running attention's KV reads (1),
the next layer's weight prefetch (1), embedding gather (1), MoE
dispatch (1), output writeback (1), and 3 spare for overlap.

---

## 9. Memory hierarchy

Three tiers, all software-managed, no coherence:

| Tier | Capacity | Location | Latency | Bandwidth | Source of sizing |
|---|---|---|---|---|---|
| **L0** (regs) | 16 KiB / tile | per Tensor Tile | 1 cycle | 4 TB/s/tile | MAC array width |
| **L1 SRAM** | 256 KiB / tile | per Tensor Tile | 3–5 cycles | 1 TB/s/tile | `gemm.md` panel + attention online-softmax |
| **L2 SRAM** | 8 MiB shared | central | ~30 cycles | 500 GB/s | layer working set + KV block × 2 streams |
| **HBM3e** (server) | 96 GiB (× 4 stacks) | external | ~150 ns | 4 TB/s | KV cache + weight pool |
| **LPDDR5X** (edge) | 16 GiB | external | ~80 ns | 100 GB/s | edge target |

### 9.1 L2 sizing rationale

8 MiB satisfies the hot working sets identified by Phase 8:

- One layer's `(γ, β)` for norms: ~16 KiB.
- Active attention's Q tile + 2 streamed K/V blocks: ~96 KiB.
- ALiBi / tree mask tile (server worst case): ~520 KiB.
- MoE dispatch buffer at batch=64: ~2 MiB.
- Full RMSNorm-fused-RoPE residual stream tile: ~256 KiB.
- Headroom for L1 spill: rest.

### 9.2 Coherence

**None.** Phase 8 §8.4 states explicitly: explicit, software-managed,
no full cache coherence. Every DMA is initiated by the compiler.
Bank conflicts in L1 are visible to the perf model and resolved by
schedule. The cost of coherent caches (area + complexity) is bigger
than the win on a workload this regular.

---

## 10. NoC (on-chip interconnect)

A **2D mesh** of bidirectional links connecting:

- N Tensor Tiles (corners + edges of the mesh).
- The Attention Engine (centre node, full bisection bandwidth to
  every tile).
- The KV Controller (adjacent to the Attention Engine, with a
  dedicated short-link).
- The Permutation Engine (centre, near L2).
- The Scan Engine instances (one per tile, no cross-tile traffic).
- L2 SRAM banks (edge-attached).

### 10.1 Bandwidth

Each link: 256 bytes/cycle bidirectional. At 1 GHz that's 256 GB/s
per link.

### 10.2 Topology choice

A mesh (rather than crossbar) absorbs MoE's all-to-all dispatch
without quadratic area cost. The Permutation Engine sits at the
centre to minimise hop count for the typical pattern (every tile
sends a few rows, every tile receives a few rows).

### 10.3 Multicast

A **multicast bit per packet** lets weight broadcasts (the
`B feeder`'s panel load) propagate to every tile in one transaction
— critical for batched-GEMM and grouped-GEMM workloads where every
tile uses the same B panel.

---

## 11. Hardware Scheduler

Sits at the chip's instruction port. Receives macro-instructions
from the host and issues them to the right blocks.

### 11.1 Responsibilities

1. **Decode** — turn each macro-instruction into a per-block
   micro-program.
2. **Resource arbitration** — track which Tile / Engine is busy;
   issue the next instruction when its operands are ready.
3. **DMA orchestration** — chain DMA descriptors so weight
   prefetches overlap with compute.
4. **Continuous-batching support** — accept `(request_id, step_id)`
   tags so multiple requests' decode steps share Tensor Tile time
   (matches the Phase 3 scheduler's expectations).

### 11.2 Why static

Phase 8's roofline analysis doesn't show meaningful workload
variability *within* a kernel — the same shape executes the same way
every time. Out-of-order issue at the hardware level wouldn't add
parallelism the compiler can't already extract. **The Scheduler is a
dispatch unit, not a CPU front-end.**

---

## 12. Edge profile vs Server profile

Same ISA, same kernel implementations. Different resource counts:

| Parameter | Edge (Llama-3-8B Q4) | Server (Llama-3-70B INT4) |
|---|---|---|
| Tensor Tiles | 1 | 8 |
| Tensor Unit MACs | 32 × 32 | 64 × 64 (per tile) |
| L1 SRAM | 128 KiB | 256 KiB (per tile) |
| L2 SRAM | 1 MiB | 8 MiB |
| Memory | LPDDR5X 16 GiB @ 100 GB/s | HBM3e 96 GiB @ 4 TB/s |
| KV Controller streams | 1 | 4 |
| Attention Engines | 1 | 1 (shared) |
| DMA channels | 2 | 8 |
| Target perf | ≥30 tok/s @ <5 W | ≥1000 tok/s @ <500 W |

Both targets are derived from the Phase 8 dossiers' bandwidth
predictions. The edge profile fits a single die ~50 mm²; the server
profile is ~600 mm² with 4× HBM3e stacks. Both run the same compiler
output verbatim; only the resource counts in the device-info block
change.

---

## 13. Numerical contracts

Anchored in Phase 1's "FP32 is the source of truth" rule and the
per-dtype tolerance policy in `tests/numerical/ulp.hpp`:

- **All accumulators FP32 or INT32.** No exceptions. The MAC array
  decisions in §3.2 enforce this.
- **All reduction lanes FP32.** Norms, softmax, online-softmax,
  scan — every reduction is precise.
- **Storage / multiplier dtypes** chosen per kernel:
  - Weights: BF16 / FP16 / FP8 / INT8 / INT4 (per-model choice).
  - Activations: BF16 / FP16 / FP8.
  - KV cache: FP16 / FP8 / INT8 / INT4 (per-deployment choice).
  - LM-head logits + sampling: FP32 (precision matters at the tail).

Bit-exact equivalence with the LonghornAI reference C++ kernels is
enforced by the validation harness: every kernel's silicon path must
produce the same output as `*_ref` within the documented tolerance,
on the same input.

---

## 14. Summary: what Phase 8 dictated

Every block in this document is justified by a Phase 8 dossier:

| Phase 8 dossier | Phase 9 block | Justification |
|---|---|---|
| `gemm.md`, `gemm_quant.md` | Tensor Unit + Rescale | compute-bound, dequant-on-feeder pattern |
| `normalization.md` | Norm Engine | dedicated block; Welford one-pass |
| `softmax.md` | Softmax Engine | shared with attention |
| `reduction.md`, `sampling.md` | Reduction Engine + PRNG lane | top-k + multinomial |
| `activation.md` | Vector Unit + GEMM epilogue | fused MLP block |
| `attention.md`, `kvcache.md` | Attention Engine + KV Controller | the silicon's load-bearing pair |
| `moe.md` | Permutation Engine + grouped-GEMM dispatch | bandwidth-limited weight reads |
| `ssm.md`, `rwkv.md`, `linear_attn.md` | Scan Engine | shared per-channel recurrence |
| `transform.md`, `embedding.md` | DMA engines | pure data motion |
| `speculative.md` | Vector Unit + PRNG lane | small math, big enable |
| `positional.md`, `rope.md` | Vector Unit (epilogue), Host setup | small / cold-path |

The companion documents — [`isa.md`](isa.md), [`roadmap.md`](roadmap.md),
[`perf_model.md`](perf_model.md) — extend this into a delivery plan.
