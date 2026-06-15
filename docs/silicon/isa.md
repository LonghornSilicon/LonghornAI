# Longhorn ISA — Tile-Granular Coarse Instruction Set

Companion to [`architecture.md`](architecture.md). The Longhorn ISA is
**coarse-grained**: one instruction issues a tile-of-work to one or
more blocks. The compiler is expected to extract instruction-level
parallelism statically; the silicon does no out-of-order issue.

This is a sketch — exact bit assignments are an RTL-team decision —
but the operation set, encoding shape, and execution model are fixed
here.

---

## 1. Encoding

128-bit fixed-width macro-instructions. Layout:

```
 127           120 119           104 103              80 79             40 39                0
┌────────────────┬────────────────┬───────────────────┬─────────────────┬───────────────────┐
│  OPCODE  (8)   │  TARGET  (16)  │  RESOURCE   (24)  │  IMM/PTR  (40)  │   FLAGS    (40)   │
└────────────────┴────────────────┴───────────────────┴─────────────────┴───────────────────┘
```

- **OPCODE**: which engine does what.
- **TARGET**: which Tile / Engine instance receives this (mesh address).
- **RESOURCE**: tile-local registers / SRAM addresses for inputs and
  output. 8 bits each for src0, src1, dst.
- **IMM/PTR**: shape constants or DMA descriptor pointers.
- **FLAGS**: per-opcode mode bits (dtype, mask mode, predicate, etc.).

A **VLIW bundle** is `{n_tiles + n_engines}` instructions issued in
the same cycle, scheduled by the compiler. Bundles are at most 16
instructions wide (matches the max chip topology of 8 Tensor Tiles +
the dedicated engines).

---

## 2. Opcode groups

### 2.1 Tensor Unit

| Opcode | Operation | Engine | Backed by |
|---|---|---|---|
| `GEMM_TILE` | `MR × NR` block, FP32 acc | Tensor Unit | `gemm` |
| `GEMM_TILE_W8A8` | INT8 × INT8, INT32 acc, FP32 rescale | Tensor Unit + feeders | `gemm_w8a8` |
| `GEMM_TILE_W4A16` | INT4 unpack on B feeder, FP32 acc | Tensor Unit + feeders | `gemm_w4a16_groupwise` |
| `GEMM_TILE_FP8` | FP8 × FP8, FP32 acc | Tensor Unit + feeders | `gemm_fp8_e4m3` |
| `GEMM_GROUP` | dispatch list of `(M_g, N_g, K_g)` ptrs | Tensor Unit + grouped feeder | `gemm_grouped` (MoE) |

The dtype of the inputs is in **FLAGS**; the same opcode covers the
matrix-shape choice and the rescale path.

### 2.2 Vector / Norm / Softmax

| Opcode | Operation | Engine |
|---|---|---|
| `VEC_OP` | pointwise add/mul/abs/clamp/etc., dtype in FLAGS | Vector Unit |
| `VEC_TRANSCEND` | exp / log / tanh / erf / rsqrt / sigmoid | Vector Unit |
| `VEC_ACTIVATION` | fused GELU / SiLU / SwiGLU / GeGLU / ReLU / Tanh | Vector Unit (or Tensor Tile epilogue) |
| `RMSNORM_ROW` | one row, with optional fused-RoPE epilogue | Norm Engine |
| `LAYERNORM_ROW` | Welford one-pass, with γ + β | Norm Engine |
| `SOFTMAX_ROW` | streaming online softmax | Softmax Engine |
| `LOG_SOFTMAX_ROW` | numerically stable log-softmax | Softmax Engine |

### 2.3 Reduction / Sampling

| Opcode | Operation | Engine |
|---|---|---|
| `REDUCE_ROW` | sum / max / mean (FLAGS pick) | Reduction Engine |
| `ARGMAX_ROW` | strict-greater compare; lowest-index tie-break | Reduction Engine |
| `TOPK_ROW` | length-`k` heap (k ≤ 64 fast path) | Reduction Engine |
| `SAMPLE_ROW` | full pipeline: T → top-k → top-p → softmax → multinomial | Vector Unit + PRNG + Reduction |

### 2.4 Attention

| Opcode | Operation | Engine |
|---|---|---|
| `ATTN_SDPA` | full materialised path (small Lk only) | Attention Engine |
| `ATTN_FLASH_PREFILL` | Q-outer, online softmax | Attention Engine |
| `ATTN_FLASH_DECODE` | split-K + cross-split merge | Attention Engine |
| `ATTN_CROSS` | Q from one stream, K/V from another | Attention Engine |
| `ATTN_PAGED` | reads K/V via the KV Controller | Attention Engine + KV Controller |

`FLAGS` encode causal / window / chunk / bias-mode (none / per-head /
shared / tree).

### 2.5 KV Cache Controller

| Opcode | Operation | Engine |
|---|---|---|
| `KV_APPEND` | write new K/V into block-table position | KV Controller |
| `KV_APPEND_Q8` | INT8 append + per-row scale | KV Controller |
| `KV_BLOCK_ALLOC` | host hint to bump refcount on a physical block | KV Controller |
| `KV_BLOCK_FREE` | host hint to drop refcount; may free | KV Controller |

### 2.6 MoE / Permutation

| Opcode | Operation | Engine |
|---|---|---|
| `MOE_DISPATCH` | counting-sort scatter, builds offsets + scatter buffer | Permutation Engine |
| `MOE_COMBINE` | weighted gather back to per-token outputs | Permutation Engine |
| `TRANSPOSE_TILE` | 2D transpose / N-D permute | Permutation Engine (or DMA) |
| `GATHER_ROWS` | indexed row gather | Permutation Engine |
| `SCATTER_ADD_ROWS` | atomic-add scatter | Permutation Engine |

### 2.7 Scan Engine

| Opcode | Operation | Engine |
|---|---|---|
| `SCAN_SSM` | Mamba S6 step, mode = SSM | Scan Engine |
| `SCAN_SSD_CHUNK` | Mamba-2 cross-chunk carry; within-chunk goes to Tensor Unit | Scan Engine + Tensor Unit |
| `SCAN_WKV` | RWKV log-space step | Scan Engine |
| `SCAN_LINATTN` | linear attention rank-1 + readout | Scan Engine |
| `CONV1D_DEPTHWISE` | length-K causal FIR per channel | Vector Unit |

### 2.8 DMA

| Opcode | Operation | Engine |
|---|---|---|
| `DMA_LINEAR` | strided streaming, descriptor in IMM | DMA channel |
| `DMA_TRANSPOSE` | transposed-tile read/write | DMA channel |
| `DMA_GATHER` | scatter-gather descriptor list | DMA channel |
| `DMA_BARRIER` | wait for all in-flight DMAs on this channel | DMA channel |

### 2.9 Control

| Opcode | Operation |
|---|---|
| `LOOP_BEGIN` | small-loop counter (max iter in IMM) |
| `LOOP_END` | branch back to `LOOP_BEGIN` |
| `PRED_SET` | set predicate from a register comparison |
| `BARRIER` | tile-local or chip-wide; FLAGS pick scope |
| `NOP` | filler in VLIW bundles |

No general indirect branches. The compiler unrolls or specialises
control flow; the only control structure on-chip is the bounded
`LOOP_*` pair (driven by a 16-bit counter in IMM).

---

## 3. Execution model

### 3.1 Static, predicated, in-order

Every instruction's start cycle is determined at compile time. The
hardware Scheduler tracks resource readiness (Tile busy / Engine busy
/ DMA in flight) and **stalls the entire bundle** if any operand
isn't available — but it does not reorder across bundles. The cost of
a stall is visible to the perf model so the compiler can schedule
around it.

### 3.2 Predication

A 4-bit predicate register file per tile. `PRED_SET` writes; opcodes
that take a predicate (any tile-local op) read it. Used for
sequence-length variability (e.g., a partial last K-tile in
attention).

### 3.3 Memory model

- **No coherence.** Loads / stores are serialised by `BARRIER`s the
  compiler emits explicitly.
- **All DMAs are explicit.** No implicit prefetch. The KV Controller's
  internal prefetcher is the one exception, and it's bounded to the
  one logical block past the current attention step.
- **L1 SRAM banks are software-allocated.** The compiler chooses
  which bank holds A panel, B panel, C tile, etc., and is responsible
  for ensuring conflicts are absent. The `bank_conflict` perf-model
  counter exposes mistakes.

### 3.4 Reproducibility

Same input + same instruction stream → same output bits. Reductions
use a fixed reduction tree; FP32 accumulators are deterministic.
This is a hard requirement — the LonghornAI test suite enforces
bit-exact equivalence between the C++ reference and any future RTL
implementation.

---

## 4. Programming model

The compiler produces a `.lhprog` file:

```
{
  device_info: { tiles: 8, l1_per_tile: 262144, l2: 8388608, ... },
  weight_pool: <DMA descriptors for HBM weight residency>,
  kv_layout:   <PagedCacheLayout per layer>,
  bundles: [
    { bundle_idx: 0, instructions: [...] },
    { bundle_idx: 1, instructions: [...] },
    ...
  ]
}
```

The host runtime owns request queueing, tokenizer state, and the
KV-cache `CacheManager` (matches the Phase 3 / 6 scheduler). Each
decode step issues the bundle list verbatim to the chip.

A **future LonghornAI compiler** (out of scope for Phase 9) lowers
the model graph (Phase 0 model runner) onto this format using the
Phase 8 kernel dossiers to pick the right opcode + engine + dtype per
operation.
