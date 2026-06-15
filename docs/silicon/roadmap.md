# Longhorn Implementation Roadmap

Three stages, each gated by the previous: **A** (analytical /
cycle-accurate simulator) → **B** (FPGA prototype) → **C** (ASIC
tape-out). Every stage validates against the LonghornAI C++ kernel
suite as the bit-exact ground truth.

---

## Stage A — Software simulator (year 1)

### A.1 Goal

A model that takes a `.lhprog` file (per [`isa.md`](isa.md)) and
produces:

1. **Bit-exact outputs** — using the C++ kernel implementations
   directly. The simulator is, fundamentally, a sequencer that
   dispatches instructions to the existing kernels.
2. **Predicted cycle counts and bandwidth utilisation** — using the
   analytical performance model in [`perf_model.md`](perf_model.md).
3. **A trace** in the same Chrome-tracing JSON format as
   `src/core/tracing.hpp`.

### A.2 Deliverables

- `tools/perf_model/perf_model.{hpp,cpp}` — the analytical model
  (this Phase 9 commit).
- `tools/perf_model/silicon_sim_main.cpp` — CLI that runs the bench
  shape suite through the model and emits predicted CSV (this commit).
- `tests/test_perf_model.cpp` — asserts bound classifications match
  every Phase 8 dossier (this commit).

### A.3 Acceptance gate

- Simulator's bound classification matches all Phase 8 dossiers.
- For every kernel, predicted execution time tracks the roofline
  prediction within the model's documented assumptions (linear
  speedup with lane count, no contention beyond L1 bank conflicts).

### A.4 Out-of-scope at Stage A

- Pipeline stalls, queue back-pressure, NoC routing — any micro-
  effect smaller than the dominant memory/compute roof. These
  surface at Stage B.

---

## Stage B — FPGA prototype (year 1.5–2)

### B.1 Goal

A working RTL implementation of the **Edge profile** on a single
mid-range FPGA (Xilinx UltraScale+ or Intel Stratix 10 class).
Sufficient to run a small decoder LLM (Llama-3-8B Q4 / W4A16) at
≥ 5 tokens/sec end-to-end.

### B.2 Deliverables

- SystemVerilog of: 1× Tensor Tile (32×32 MACs), 1× Attention Engine,
  1× KV Controller, 1× Scan Engine, 1× Permutation Engine, 2× DMA
  channels, 128 KiB L1, 1 MiB L2, LPDDR4 / LPDDR5 controller.
- Cocotb / UVM test bench that drives `.lhprog` files into the RTL
  and compares each kernel's output to the C++ reference.
- A `cycle_accurate.so` build of the simulator that wraps the RTL via
  Verilator; replaces the analytical perf model for accurate timing.
- A bring-up paper trail mapping each block back to its Phase 8
  dossier and Phase 9 §3–7 spec.

### B.3 Bring-up milestones

| # | Milestone | Validation |
|---|---|---|
| B-1 | Tensor Tile alone runs `gemm` (FP16 + BF16 + INT8 + INT4) | All dtypes match `gemm_*` C++ refs bit-exactly |
| B-2 | Tensor Tile + Norm + Softmax + Vector chain | LayerNorm, RMSNorm, softmax, GELU/SiLU all match |
| B-3 | Attention Engine + KV Controller integrated | `flash_attention` + `paged_attention` match SDPA |
| B-4 | Scan Engine integrated | Mamba `selective_scan` + RWKV `wkv` match |
| B-5 | Permutation Engine + grouped-GEMM dispatch | `moe_forward` matches per-token reference |
| B-6 | Scheduler + multi-bundle issue | A 4-layer Llama block passes a forward parity test |
| B-7 | Full Llama-3-8B Q4 forward, 1 token | Logits match the reference C++ run within FP tolerance |
| B-8 | Llama-3-8B Q4 decode at ≥ 5 tok/s | End-to-end perf gate |

### B.4 Acceptance gate

- B-7 passes (numerical correctness end-to-end).
- B-8 passes (performance target met on the FPGA target's bandwidth).
- The cycle-accurate simulator's predicted timing matches FPGA
  measurement within 10% on the kernel suite.

---

## Stage C — ASIC tape-out (year 2.5+)

### C.1 Sequencing

**Edge profile first.** Smaller die, lower-risk physical-design
exercise, gives a stand-alone product. Server profile follows once
the Edge profile is silicon-validated.

| Profile | Tape-out | Process | Die area | TDP target | Models |
|---|---|---|---|---|---|
| Edge | C-1 | 7 nm | ~50 mm² | < 5 W | Llama-3-8B Q4, Mistral 7B Q4 |
| Server | C-2 | 5 nm | ~600 mm² | ~500 W | Llama-3-70B INT4, Mixtral, DeepSeek-MoE |

### C.2 Pre-silicon validation

Before each tape-out:

1. Full LonghornAI test suite (currently 117 tests) passes on the RTL
   via cycle-accurate simulation. Bit-exact kernel parity, runtime
   acceptance gates (continuous batching ≥ 80% occupancy, MoE
   grouped-GEMM ≥ 70% of dense, etc.) all met against the analytical
   prediction.
2. Five model-shape regression suites (Llama-3-8B, Llama-3-70B,
   Mistral-7B, Mixtral-8×7B, RWKV-v6) generate logit traces; the RTL
   must reproduce those traces bit-exactly.
3. Roofline match within 10% across every kernel in the bench suite.

### C.3 Post-silicon

- Bring-up board with HBM3e × 4 (server) or LPDDR5X × 1 (edge).
- The same `.lhprog` files that ran on FPGA run unmodified on ASIC
  bring-up — the ISA contract is the silicon's external surface.
- A single firmware update path lets the Hardware Scheduler's
  micro-ops evolve without a re-spin.

---

## Software stack milestones (parallel to A/B/C)

The architecture is useless without the compiler that targets it.
LonghornAI is the *kernel* foundation; the compiler is a separate
project but tracks the same milestones.

| Milestone | What it means |
|---|---|
| S-1 | LonghornAI compiler MVP — lowers a hand-coded Llama block to `.lhprog` |
| S-2 | Auto-tuner picks per-kernel `(MR, NR, KR)` from the perf model |
| S-3 | Quantization pipeline (AWQ / GPTQ / GGUF Q4_K_M) ingests checkpoints |
| S-4 | Full Llama-3 / Mistral / Qwen graph compilation |
| S-5 | MoE graph (Mixtral, DeepSeek-MoE) compilation |
| S-6 | Mamba / RWKV / Jamba (hybrid) compilation |
| S-7 | Production serving runtime: continuous batching, prefix cache, speculative decoding (Phase 6 features wired through) |

S-1 through S-3 should land alongside Stage B; S-4 through S-7
support Stage C and the server-profile launch.

---

## Risk register and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| HBM bandwidth doesn't materialise at 4 TB/s sustained | High | KV Controller's prefetcher is the load-bearing fallback; W4A16 / Q8 KV are the bandwidth multipliers; perf model exposes the gap so we know early. |
| MoE all-to-all on multi-die designs needs more NoC than budgeted | Medium | Single-die first; multi-die only if the workload demands and after measuring real Mixtral / DeepSeek runs on the simulator. |
| Long-context (1M tokens) attention doesn't fit single-chip KV pool | Medium | Sparse-KV (H2O), MLA (DeepSeek), and FlashDecoding all in scope; pick one before tape-out based on which is most validated by Stage A simulator runs. |
| New post-Transformer architecture (post-2026) doesn't match the engine inventory | Low–Medium | Mamba-2 SSD already shares ≥ 80% of attention's tensor-unit code. The Scan Engine is the hedge for everything that isn't attention-shaped. |
| Quantization fidelity collapses for new instruction-tuned models | Medium | Default deployment dtype = FP8 (E4M3); INT4 is opt-in per model with perplexity gates from a calibration suite (S-3). |
