# LonghornAI — Engineering Roadmap & Architecture Plan

> **Status:** Draft v1.0 · **Owner:** Principal GPU/AI Software Architecture
> **Audience:** Architecture, Software, Compiler, Performance, and Silicon teams
> **Review cadence:** Quarterly (aligned to milestone gates)
> **Repository:** `longhorn-silicon/longhornai`

---

## Table of Contents

1. [Vision](#1-vision)
2. [Repository Architecture](#2-repository-architecture)
3. [Kernel Roadmap](#3-kernel-roadmap)
4. [Benchmarking Strategy](#4-benchmarking-strategy)
5. [Validation Infrastructure](#5-validation-infrastructure)
6. [Performance Engineering](#6-performance-engineering)
7. [Model Support Matrix](#7-model-support-matrix)
8. [Milestone Schedule](#8-milestone-schedule)
9. [Success Metrics](#9-success-metrics)
10. [Future Vision](#10-future-vision)

---

## 1. Vision

LonghornAI will become the **canonical AI inference kernel library and runtime substrate for all Longhorn Silicon accelerators** — the single source of truth from which every inference workload, benchmark, and silicon validation suite is derived. It plays, for Longhorn, the combined role that CUTLASS, cuBLAS, FlashAttention, TensorRT-LLM, and vLLM collectively play for the NVIDIA ecosystem, but is architected from day one to be **hardware-portable across the full Longhorn execution-target lifecycle**: CPU functional models, RTL co-simulation, FPGA prototypes, silicon bring-up boards, and production accelerators.

### 1.1 Strategic Goals

- **Canonical kernel authority.** Every operator that touches a Longhorn accelerator — from GEMM to Paged Attention to MoE dispatch — has exactly one reference implementation, one numerical golden, and one performance contract that lives in this repository. Downstream products (compiler, serving runtime, model toolkit) consume LonghornAI; they do not fork it.
- **Hardware/software co-design loop.** LonghornAI is the primary vehicle through which kernel and silicon teams iterate together. Microarchitectural decisions (tensor-core shapes, scratchpad sizing, memory hierarchy, interconnect topology) are validated against *real workloads* in this repo before tapeout commitments.
- **Bring-up acceleration.** When first silicon arrives, the validation workloads, golden references, and performance baselines required to qualify the part already exist and have been exercised against RTL and FPGA. Silicon bring-up is a *re-targeting* exercise, not a from-scratch build.
- **Performance leadership.** Longhorn kernels target competitive parity with, and on favorable workloads superiority over, the best-in-class CUDA ecosystem implementations — measured rigorously, not aspirationally.
- **Portability without abstraction tax.** A backend abstraction layer lets the same operator definitions target multiple ISAs and simulation environments, but the architecture must never force a lowest-common-denominator implementation that sacrifices peak performance on the production target.

### 1.2 Non-Goals (initial phases)

- Training kernels (forward/backward, optimizer states). LonghornAI is **inference-first**; training support is explicitly out of scope through at least the 24-month horizon.
- A general-purpose tensor framework competing with PyTorch/JAX. We integrate with them; we do not replace them.
- Graph-level model compilation in Phase 1–3 (this becomes the Longhorn Compiler Stack, seeded later — see §10).

---

## 2. Repository Architecture

```
longhornai/
├── kernels/            # Core compute kernels (the heart of the repo)
├── runtime/            # Execution runtime, scheduler, memory mgmt, backends
├── benchmarks/         # Performance benchmarking harness & comparison suites
├── validation/         # Correctness, numerical, RTL/FPGA/silicon validation
├── models/             # End-to-end model graphs & support definitions
├── quantization/       # Quantization algorithms, calibration, packed formats
├── compiler/           # Kernel codegen, auto-tuning, scheduling primitives
├── docs/               # Architecture, ABI, kernel authoring guides, ADRs
├── scripts/            # Build, CI, profiling, target-bringup automation
└── tests/              # Cross-cutting integration & regression tests
```

### 2.1 Directory Responsibilities

#### `kernels/`
The core of the repository. Hardware-portable kernel implementations organized by operator family, each with a target-independent reference and one or more target-specialized backends.

```
kernels/
├── gemm/               # GEMM, batched, grouped, tensor contractions
├── normalization/      # LayerNorm, RMSNorm
├── activation/         # GELU, SiLU, fused activations
├── attention/          # SDPA, FlashAttn v1/v2, MHA/MQA/GQA, paged
├── moe/                # Router, gating, dispatch, expert parallel
├── elementwise/        # Reductions, softmax, RoPE, embedding lookup
├── quant/              # Dequant/quant fast paths, packed-format matmul
├── fused/              # Cross-family fusions (e.g. RMSNorm+QKV proj)
├── common/             # Tiling, iterators, predication, swizzle utilities
└── backends/           # Per-target lowering: cpu/, sim/, rtl/, fpga/, lhsil/
```

Each kernel directory carries: (1) a `reference/` implementation in portable C++/Python that defines numerical semantics, (2) `impl/` backend-specialized versions, (3) `tuning/` config spaces, and (4) a `KERNEL.md` contract specifying inputs, layouts, supported dtypes, and the performance target.

#### `runtime/`
The thin, low-overhead execution layer that dispatches kernels, manages device memory, sequences operators, and exposes the LonghornAI ABI to host frameworks. Contains the backend dispatch registry, stream/queue abstractions, the KV-cache allocator, paged memory manager, and the continuous-batching scheduler. This is what a serving stack links against.

#### `benchmarks/`
Reproducible performance measurement. Houses microbenchmarks (single-kernel), operator-suite benchmarks, and end-to-end model benchmarks, plus the comparison harness that runs identical workloads against CUTLASS/cuBLAS/TensorRT-LLM/FlashAttention/Triton/vLLM and emits normalized reports (throughput, latency, % of reference peak, roofline position). All results are versioned and tracked to detect regressions.

#### `validation/`
Correctness across the entire target lifecycle. Unit and numerical tests, golden-reference generation/storage, differential testing against framework references, and the cross-target equivalence harness that proves CPU-model ≡ RTL ≡ FPGA ≡ silicon within defined tolerances. Contains the silicon bring-up validation playbooks.

#### `models/`
End-to-end model definitions assembled from LonghornAI kernels (Llama, DeepSeek, Qwen, Gemma, Mistral, Phi, Mixtral). Each model has a graph definition, a weight-loading/conversion path, a config-to-kernel mapping, and an accuracy harness (perplexity, task evals) used to certify that the kernel composition produces correct generation, not just correct individual ops.

#### `quantization/`
Quantization algorithm implementations (INT8, INT4, AWQ, GPTQ, SmoothQuant, activation quantization), calibration pipelines, packed weight-format definitions, and the dtype/format ABI shared with `kernels/quant/`. Separates the *algorithm* (how to derive quantized weights) from the *kernel* (how to compute on them).

#### `compiler/`
Kernel-level code generation and tuning infrastructure (not yet whole-graph compilation). Tiling/schedule DSL, the auto-tuning search engine, template instantiation, and target lowering hooks. This is the seed crystal for the future Longhorn Compiler Stack (§10).

#### `docs/`
Architecture decision records (ADRs), the kernel ABI specification, kernel-authoring guides, the backend porting guide, numerical-tolerance policy, and the performance contract methodology. Documentation is a first-class deliverable, reviewed with code.

#### `scripts/`
Automation: build orchestration across targets, CI entry points, profiler wrappers, golden regeneration, FPGA bitstream flows, and silicon bring-up board provisioning. Keeps target-specific glue out of source directories.

#### `tests/`
Cross-cutting integration and regression tests that span multiple directories — full decode loops, scheduler correctness under continuous batching, multi-device communication, and the regression gates that block merges on numerical or performance drift.

### 2.2 Backend Abstraction & Target Lifecycle

A central design principle: kernels are authored against a **target-independent semantic reference** and lowered through `kernels/backends/` to each execution target. The supported targets, in maturity order:

| Target | Path | Purpose | Fidelity |
|--------|------|---------|----------|
| CPU functional model | `backends/cpu` | Fast iteration, golden generation, CI default | Bit-exact numerics, no timing |
| Architectural simulator | `backends/sim` | Performance modeling, occupancy/cycle estimates | Cycle-approximate |
| RTL co-simulation | `backends/rtl` | Pre-silicon HW/SW validation | Cycle-accurate, slow |
| FPGA prototype | `backends/fpga` | Pre-silicon at-scale validation | Real timing, reduced clock |
| Longhorn Silicon | `backends/lhsil` | Production accelerator | Full performance target |

The same operator test runs against every available backend; the cross-target equivalence harness (§5) enforces that they agree.

---

## 3. Kernel Roadmap

Kernels are sequenced so that each phase unlocks the next: foundational compute enables attention, attention enables decode, decode enables quantization/MoE optimization at the workloads that matter. Every kernel ships with a reference, golden, tuning space, and performance contract before it is considered "done."

### Phase 1 — Foundational Compute Kernels

The bedrock. Nothing downstream is credible until GEMM and the elementwise/normalization primitives hit their performance contracts.

| Kernel | Notes / Acceptance |
|--------|--------------------|
| **GEMM** | Core tensor-core path; FP16/BF16/FP8 in, FP32 accum. Tiled, double-buffered, target ≥ cuBLAS-class efficiency on key shapes. |
| **Batched GEMM** | Uniform-shape batching for MHA projections and small-matrix regimes. |
| **Grouped GEMM** | Variable-shape grouping — prerequisite for MoE expert matmuls and ragged batches. |
| **Tensor contractions** | Generalized einsum-style contractions over the GEMM core. |
| **LayerNorm** | Fused mean/variance, welford-stable; FP32 reduction. |
| **RMSNorm** | Llama/Qwen/Mistral norm; fuse-ready with subsequent projection. |
| **Softmax** | Online/streaming-stable; the numerical kernel attention depends on. |
| **GELU** | tanh and erf variants; exact-match golden to framework. |
| **SiLU** | SwiGLU activation path for gated MLPs. |
| **RoPE** | Rotary embeddings; interleaved & half-rotation layouts, NTK/linear scaling. |
| **Embedding lookup** | Gather with optional scaling; large-vocab friendly. |
| **Reduction kernels** | Sum/max/mean primitives backing norms and softmax. |

**Exit gate (Phase 1):** GEMM within target % of reference peak on the shape suite; all elementwise/norm kernels bit-compatible with goldens within tolerance; tuning framework operational for GEMM.

### Phase 2 — Attention Kernels

| Kernel | Notes / Acceptance |
|--------|--------------------|
| **Scaled Dot-Product Attention** | Reference fused attention; correctness anchor for the flash variants. |
| **FlashAttention v1** | IO-aware tiled attention, no materialized score matrix; numerical parity with reference. |
| **FlashAttention v2** | Improved work partitioning & warp scheduling; the prefill performance target. |
| **Multi-Head Attention** | Full-head path with fused QKV and output projection. |
| **Multi-Query Attention** | Single KV head; KV-cache footprint reduction. |
| **Grouped-Query Attention** | Llama-3/Qwen2 GQA; configurable KV-head grouping. |
| **KV Cache kernels** | Append, gather, layout management; FP16/FP8/INT8 cache dtypes. |
| **Paged Attention** | Block-table indirection over paged KV; the foundation for high-throughput serving. |

**Exit gate (Phase 2):** FlashAttention v2 parity (latency & numerics) with the reference CUDA implementation on the benchmark causal/non-causal suite; Paged Attention correct under fragmented block tables.

### Phase 3 — LLM Decode Kernels

Where kernels become a serving system.

| Capability | Notes / Acceptance |
|-----------|--------------------|
| **Prefill path** | Long-context, compute-bound; FlashAttention v2 + fused projections. |
| **Decode path** | Single-token, memory-bound; optimized for KV-cache bandwidth and low latency. |
| **Continuous batching** | Iteration-level scheduling; insert/evict requests without draining the batch. |
| **Speculative decoding** | Draft-then-verify; target-model batched verification kernels. |
| **Dynamic batching** | Adaptive batch assembly under latency SLOs. |
| **Prefix caching** | Shared-prefix KV reuse across requests; block dedup in the paged allocator. |

**Exit gate (Phase 3):** End-to-end Llama decode loop runs under continuous batching on the architectural simulator/FPGA with correct generation and a published tokens/sec baseline.

### Phase 4 — Quantization Kernels

| Technique | Notes / Acceptance |
|-----------|--------------------|
| **INT8** | W8A8 and W8A16 GEMM paths; per-tensor & per-channel scales. |
| **INT4** | W4A16 weight-only; packed-format matmul with fused dequant. |
| **AWQ** | Activation-aware weight quantization; calibration pipeline + kernel. |
| **GPTQ** | Second-order weight quantization; group-wise scales/zeros. |
| **SmoothQuant** | Activation-outlier migration enabling W8A8 accuracy. |
| **Activation quantization** | Dynamic & static activation quant; fused into GEMM epilogues/prologues. |

**Exit gate (Phase 4):** INT4 weight-only and INT8 W8A8 paths meet accuracy targets (≤ defined perplexity delta) at a measured speedup over the FP16 baseline.

### Phase 5 — Mixture of Experts (MoE)

| Kernel | Notes / Acceptance |
|--------|--------------------|
| **Router kernels** | Gating logits projection + normalization. |
| **Top-K gating** | Top-1/Top-2/Top-K selection with renormalized weights. |
| **Expert dispatch** | Permutation/scatter of tokens to experts; ties into Grouped GEMM. |
| **Expert parallelism** | Sharding experts across devices; all-to-all token exchange. |
| **Token routing** | Combine/un-permute path; capacity handling & drop policy. |

**Exit gate (Phase 5):** Mixtral and DeepSeek-MoE decode correct end-to-end; expert-parallel all-to-all functional across multiple devices.

### Phase 6 — Advanced Features

| Feature | Notes / Acceptance |
|---------|--------------------|
| **Sparse attention** | Sliding-window, block-sparse, and dilated patterns. |
| **Structured sparsity** | 2:4 / N:M weight sparsity exploiting HW sparse tensor support. |
| **Fused kernels** | RMSNorm+QKV, attention+output-proj, MoE router+dispatch fusions. |
| **Operator fusion** | General epilogue/prologue fusion framework in `kernels/fused/`. |
| **Multi-GPU communication** | All-reduce, all-gather, reduce-scatter, all-to-all collectives. |
| **Distributed inference** | Tensor-parallel, pipeline-parallel, and expert-parallel orchestration. |

**Exit gate (Phase 6):** Tensor-parallel Llama-70B-class and expert-parallel Mixtral-class inference functional across the interconnect with published scaling efficiency.

---

## 4. Benchmarking Strategy

Benchmarking is a **first-class engineering discipline**, not a post-hoc activity. Every kernel has a performance contract, and every merge is gated on regression detection.

### 4.1 Reference Baselines

We benchmark each operator family against the strongest available CUDA-ecosystem implementation, normalized to the same workload, dtype, and problem shape:

| LonghornAI Component | Primary Comparison Targets |
|----------------------|----------------------------|
| GEMM / batched / grouped | **CUTLASS**, **cuBLAS** |
| Attention (prefill/decode) | **FlashAttention v2** |
| End-to-end LLM serving | **TensorRT-LLM**, **vLLM** |
| General fused/elementwise | **Triton kernels** |

The comparison harness in `benchmarks/` runs the *identical* logical workload through both stacks, captures hardware counters, and produces a normalized report so that "% of reference" is apples-to-apples.

### 4.2 Metrics

| Metric | Definition & Use |
|--------|------------------|
| **Throughput** | Operations or requests per second at saturation. |
| **Latency** | p50 / p90 / p99 per-operation and per-token latency under load. |
| **Tokens/sec** | End-to-end generation throughput (per-request and aggregate). |
| **Memory bandwidth** | Achieved GB/s vs. peak HBM/scratchpad bandwidth (decode is BW-bound). |
| **FLOPS utilization** | Achieved vs. peak tensor-core FLOPS (prefill/GEMM are compute-bound). |
| **Roofline efficiency** | Position relative to the roofline ridge; distance from the bound. |

Each kernel's `KERNEL.md` states which metric is its *binding constraint* and the target value. Decode kernels are judged on bandwidth and tokens/sec; GEMM and prefill on FLOPS utilization and roofline efficiency.

### 4.3 Methodology

- **Warm-up + steady-state** measurement; discard cold-start; report distribution not just mean.
- **Shape suites** drawn from real model configs (head dims, hidden sizes, vocab, seq lengths from the §7 models).
- **Sweeps** across batch size, sequence length, and context fill to map the operating envelope, not a single point.
- **Roofline characterization** per kernel to classify compute- vs. memory-bound and validate that optimization effort targets the actual bottleneck.
- **Regression gating**: every benchmark result is versioned; merges that regress a contract beyond threshold are blocked.

---

## 5. Validation Infrastructure

Correctness is enforced across the entire target lifecycle — the same logical test must pass on CPU model, simulator, RTL, FPGA, and silicon.

### 5.1 Layers of Validation

- **Unit tests.** Per-kernel functional tests over shape/dtype/layout matrices, including edge cases (empty, single-element, non-aligned, max-context).
- **Numerical validation.** Tolerance-based comparison (ULP / relative / absolute, per the dtype policy in `docs/`) against high-precision references. FP32 reference, FP16/BF16/FP8 within published bounds.
- **Golden references.** Versioned, checked-in (or content-addressed) golden outputs generated from the portable reference implementations. Goldens are the contract; kernels are validated against them, not against each other.
- **Differential testing.** Randomized-input comparison between the LonghornAI kernel and an independent framework reference (PyTorch/JAX/CUDA), with shrinking on failure to a minimal reproducer.
- **RTL validation.** Kernels executed in RTL co-simulation; bit-exact (or within fixed-point tolerance) agreement with the CPU functional model. Slow but authoritative pre-silicon.
- **FPGA validation.** At-scale execution on FPGA prototypes — real timing behavior, larger workloads than RTL permits, exercises memory hierarchy and interconnect.
- **Silicon validation.** Bring-up playbooks that run the validation suite on first silicon: power-on smoke tests → per-kernel correctness → end-to-end model accuracy → performance characterization. The workloads already exist from prior phases, so bring-up is a *re-targeting and qualification* exercise.

### 5.2 Cross-Target Equivalence Harness

The keystone of the strategy: a harness that runs a given operator+input across *all available backends* and asserts agreement within the per-dtype tolerance. This is what lets the silicon team trust that a kernel which passed on RTL will behave correctly on the part — and lets the kernel team catch HW/SW divergence before tapeout.

### 5.3 Accuracy Validation (model level)

Beyond per-kernel correctness, `models/` carries end-to-end accuracy harnesses (perplexity on held-out corpora, downstream task evals) to certify that *kernel composition* produces correct generation — catching errors (e.g., RoPE/scale/layout mismatches) that pass per-op tolerance but corrupt outputs.

---

## 6. Performance Engineering

Performance is engineered against the roofline, instrumented with hardware counters, and automated where the search space is large.

### 6.1 Techniques

- **Occupancy analysis.** Model and measure concurrent work in flight per compute unit; balance against register/scratchpad limits to find the latency-hiding sweet spot.
- **Register pressure analysis.** Track per-thread/per-lane register usage; manage spills; tune tile sizes to the register budget.
- **Shared-memory / scratchpad tuning.** Size tiles and double-buffers to the on-chip memory; minimize bank conflicts; stage operands for tensor-core feed.
- **Cache optimization.** Tile for L2/last-level reuse; structure access for temporal/spatial locality; manage residency for weight reuse in decode.
- **Memory coalescing.** Ensure aligned, contiguous, vectorized accesses; choose layouts (and swizzles) that maximize achieved DRAM bandwidth.
- **Kernel fusion.** Collapse memory-bound op chains (norm→proj, attn→out-proj, router→dispatch) to cut HBM round-trips — the single highest-leverage decode optimization.
- **Auto-tuning framework.** A search engine in `compiler/` that explores tile sizes, pipeline depths, swizzles, and schedules per target, caches winning configs per (op, shape, dtype, backend), and feeds the kernel dispatch registry.

### 6.2 Methodology

Every optimization is **roofline-justified**: we first classify the kernel as compute- or memory-bound, then direct effort at the binding constraint and re-measure. Profiling wrappers in `scripts/` standardize counter collection across targets so the simulator, FPGA, and silicon profiles are comparable.

---

## 7. Model Support Matrix

Models are certified end-to-end (correctness + accuracy + a published perf baseline) per the milestone schedule. Precision columns indicate the supported/targeted quantization paths.

| Model Family | Architecture Notes | FP16/BF16 | FP8 | INT8 (W8A8) | INT4 (W4A16) | Milestone |
|--------------|--------------------|:---------:|:---:|:-----------:|:------------:|-----------|
| **Llama** (2/3) | RMSNorm, RoPE, GQA, SwiGLU | ✅ | ✅ | ✅ | ✅ | M2 (FP16) → M5 (quant) |
| **Mistral** | Sliding-window attn, GQA | ✅ | ✅ | ✅ | ✅ | M3 → M6 |
| **Qwen** (2/2.5) | GQA, RoPE scaling, large vocab | ✅ | ✅ | ✅ | ✅ | M3 → M6 |
| **Gemma** | GeGLU, RMSNorm variants, large vocab | ✅ | ◐ | ✅ | ◐ | M4 |
| **Phi** | Compact dense decoder | ✅ | ◐ | ✅ | ✅ | M4 |
| **Mixtral** | Sparse MoE (Top-2), expert parallel | ✅ | ◐ | ◐ | ◐ | M6 (MoE) |
| **DeepSeek** (V2/MoE) | MLA / fine-grained MoE, expert parallel | ◐ | ◐ | ◐ | ◐ | M7 |

Legend: ✅ planned/supported · ◐ planned, later milestone · blank = out of scope for horizon.

Each certified model entry links to: its graph definition in `models/`, its accuracy report, and its tokens/sec baseline in `benchmarks/`.

---

## 8. Milestone Schedule

A realistic 18–24 month roadmap, eight quarterly gates. Each milestone is a **gate**: it must meet its exit criteria (correctness + performance contract) before the next phase's perf work begins. Dates assume the program kicks off at Q3 2026.

| Milestone | Quarter | Theme | Primary Deliverables | Exit Gate |
|-----------|---------|-------|----------------------|-----------|
| **M1** | Q3 2026 | Foundation & infra | Repo skeleton, backend abstraction (CPU + sim), GEMM v1, elementwise/norm kernels, golden + diff-test harness, benchmark harness, CI | GEMM within initial % of cuBLAS on shape suite; all Phase-1 kernels pass goldens |
| **M2** | Q4 2026 | Compute maturity + attention start | Batched/grouped GEMM, tensor contractions, RoPE/embedding, auto-tuning v1; SDPA + FlashAttention v1; **Llama FP16 forward correct (CPU/sim)** | FlashAttn v1 numerical parity; GEMM perf contract met |
| **M3** | Q1 2027 | Attention performance | FlashAttention v2, MHA/MQA/GQA, KV-cache kernels; **Mistral + Qwen FP16 correct**; RTL backend online | FA v2 parity (latency + numerics); RTL ≡ CPU on attention |
| **M4** | Q2 2027 | Decode & serving | Paged Attention, prefill/decode split, continuous batching, dynamic batching; Gemma + Phi support; **first FPGA bring-up** | E2E Llama decode under continuous batching on FPGA; tokens/sec baseline published |
| **M5** | Q3 2027 | Quantization | INT8 W8A8, INT4 W4A16, SmoothQuant; AWQ/GPTQ calibration; prefix caching, speculative decoding | INT4/INT8 accuracy targets met at measured speedup |
| **M6** | Q4 2027 | MoE & multi-device | Router/Top-K/dispatch, Grouped-GEMM MoE, expert parallelism, collectives; **Mixtral E2E**; tensor parallelism | Mixtral decode correct; TP + expert-parallel functional |
| **M7** | Q1 2028 | Advanced + DeepSeek | Sparse/structured-sparse attention, full fusion framework, distributed inference; **DeepSeek-MoE support**; **silicon bring-up (first part)** | Silicon smoke + per-kernel correctness pass; cross-target equivalence holds |
| **M8** | Q2 2028 | Silicon performance & hardening | Silicon performance characterization, auto-tuning on silicon, regression hardening, production runtime packaging | Production inference readiness criteria met (§9) |

> Schedule risk is concentrated at the RTL→FPGA→silicon transitions (M3/M4/M7), which depend on hardware availability. The CPU/sim-first strategy is precisely what de-risks these: software is validated and performance-modeled before hardware exists.

---

## 9. Success Metrics

KPIs are measurable, tracked continuously in `benchmarks/`, and reviewed at each gate.

### 9.1 Kernel Performance KPIs

| KPI | Target |
|-----|--------|
| **GEMM % of cuBLAS** | ≥ 90% of cuBLAS-class peak on the production shape suite (FP16/BF16); ≥ 85% for FP8/INT8 paths |
| **FlashAttention parity** | ≥ 95% of FlashAttention v2 prefill throughput; ≤ +5% decode latency vs. reference |
| **Roofline efficiency** | Compute-bound kernels ≥ 80% of compute roof; memory-bound kernels ≥ 85% of bandwidth roof |
| **Quantization accuracy** | INT8 W8A8 ≤ 1% perplexity delta vs. FP16; INT4 W4A16 ≤ defined per-model bound — at ≥ measured target speedup |

### 9.2 System / Serving KPIs

| KPI | Target |
|-----|--------|
| **Tokens/sec** | Per-model aggregate decode throughput targets met on the production target at defined batch/context points |
| **Latency SLO** | p99 inter-token latency within serving SLO under continuous batching at target concurrency |
| **Scaling efficiency** | Tensor-parallel and expert-parallel scaling ≥ defined % efficiency across the interconnect |

### 9.3 Program / Readiness KPIs

| KPI | Target |
|-----|--------|
| **Silicon bring-up readiness** | 100% of bring-up validation suite authored and passing on RTL+FPGA *before* first silicon |
| **Cross-target equivalence** | 100% of certified kernels agree (within tolerance) across CPU/sim/RTL/FPGA/silicon |
| **Production inference readiness** | All §7 milestone models certified (correctness + accuracy + perf) on silicon; runtime packaged; regression gates green |
| **Coverage & regression health** | Kernel functional coverage target met; zero open numerical regressions; benchmark regressions auto-gated |

---

## 10. Future Vision

LonghornAI is the kernel and runtime substrate. As it matures, four products grow out of it — each consuming LonghornAI rather than forking it.

### 10.1 Longhorn TensorRT Equivalent (Inference Optimizer)
A graph-level inference optimization engine that ingests a trained model, applies layer/operator fusion, precision selection, and kernel auto-selection from the LonghornAI library, and emits a tuned, serializable inference engine for Longhorn Silicon. The fusion framework (Phase 6) and auto-tuner (`compiler/`) are its seeds.

### 10.2 Longhorn Compiler Stack
Evolution of `compiler/` from per-kernel codegen into a full graph compiler: a tensor IR, scheduling/tiling passes, automatic operator fusion across the whole graph, and target lowering to every backend in the lifecycle. Closes the loop from model graph → optimized kernels without hand-authoring.

### 10.3 Longhorn Serving Runtime
Productization of `runtime/` into a standalone, OpenAI-compatible inference server: continuous batching, paged KV management, prefix caching, speculative decoding, multi-device orchestration, and request scheduling under SLOs — the vLLM/TensorRT-LLM-serving analog, built natively on Longhorn kernels.

### 10.4 Longhorn Model Optimization Toolkit
Productization of `quantization/` plus pruning, distillation hooks, sparsity tooling, and calibration workflows into a user-facing toolkit that takes a model from full precision to a Longhorn-optimized, quantized, sparsified deployment artifact — with accuracy guardrails enforced by the §5 accuracy harnesses.

---

### Appendix A — Governance & Engineering Conventions

- **One reference, one golden, one contract per kernel.** No kernel merges without all three.
- **ADRs in `docs/`** for every cross-cutting architectural decision (ABI changes, layout conventions, tolerance policy).
- **Roofline-justified optimization.** Performance work must cite the binding constraint it targets.
- **Cross-target green before silicon.** No kernel is "silicon-ready" until it passes the equivalence harness on all available pre-silicon backends.
- **Quarterly gates are hard gates.** Phase N+1 performance work does not begin until Phase N exit criteria are met.

*End of PLAN.md — to be reviewed by Architecture, Software, Compiler, Performance, and Silicon leads at the next quarterly gate.*
