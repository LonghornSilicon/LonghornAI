# LonghornAI — Master Plan

**A Silicon-First Inference Kernel Library and Accelerator Specification**

---

## 0. Document Purpose

This document is the canonical engineering roadmap for **LonghornAI**, the
software companion to the future **Longhorn** transformer-inference
accelerator. It is simultaneously:

1. A **kernel library plan** — how to build, validate, and benchmark every
   inference primitive a modern decoder LLM needs.
2. A **silicon specification document** — how each kernel translates into
   datapaths, memory hierarchies, dispatch units, and on-chip interconnect.
3. A **company roadmap** — phased deliverables, milestone gates, performance
   goals, and the bridge from a portable C++ reference to ASIC tape-out.

The philosophy is taken directly from companies like Etched: **inference
workloads first, kernels second, microarchitecture third**. We do not start
with a compute fabric and look for things to run on it. We start with the
operations that real LLMs require, build clean reference implementations,
profile their arithmetic intensity and data-movement patterns, and only then
allow those numbers to constrain the silicon design.

Every kernel in this repository is therefore three artifacts at once: a
correctness oracle (the naive reference), a baseline for software performance
(the optimized CPU implementation), and a silicon work-item (the hardware
mapping notes that will inform RTL).

---

## 1. Vision

LonghornAI must eventually constitute the **complete software specification**
for Longhorn silicon. By the end of the program, the repository will contain a
sufficient set of kernels to run forward inference on every modern
transformer-class architecture, including:

- **Dense decoder LLMs:** Llama 2 / 3 / 4, Mistral, Qwen, Gemma, Phi, GPT-style.
- **Sparse / MoE models:** Mixtral, DeepSeek (V2/V3/MoE), Qwen-MoE.
- **Long-context models:** sliding-window, Mamba/SSM, hybrid Transformer-SSM
  (Jamba-style), linear-attention variants.
- **Alternative architectures:** RWKV (v4–v7), Mamba / Mamba-2, RetNet, Hyena.
- **Future decoder-only variants** that share the same kernel inventory.

The goal is not training. The goal is **world-class inference** — the lowest
latency at the lowest power for production-scale LLM serving. Training kernels
are explicitly out of scope; backward passes, optimizers, gradient
accumulation, and distributed training collectives never enter the repository.

The soft-target end-state is:

> *Any modern decoder LLM checkpoint can be loaded, sharded, quantized,
> KV-cached, and served end-to-end using kernels in this repository, with
> bit-exact equivalence to the naive references and a clearly documented
> hardware mapping for each operation.*

---

## 2. Strategic Thesis: Kernels → Microarchitecture

The repository is organized as a **funnel from software to silicon**:

```
Phase 0–2:  General-purpose kernel infrastructure and the dense-transformer
            forward pass on CPU. Establishes correctness, numerics, and the
            test/benchmark scaffolding everything else relies on.

Phase 3–6:  Inference-specific systems engineering: KV cache, quantization,
            MoE, batching/scheduling, speculative decoding. This is where
            the *workload* starts dictating the *hardware*.

Phase 7:    Alternative architectures. Forces the kernel inventory to be
            general enough to absorb non-attention recurrence kernels (SSM,
            RWKV, linear attention) without architectural retrofits.

Phase 8:    Per-kernel silicon classification — compute-bound vs. memory-bound,
            systolic-friendly vs. streaming, on-chip SRAM vs. HBM, and the
            corresponding accelerator block candidates.

Phase 9:    Synthesis. Using only the data produced by Phases 0–8, derive a
            candidate Longhorn microarchitecture, an instruction format, an
            execution model, and an FPGA-prototype roadmap.
```

This funnel is deliberate. Phase 9 is **not** a separate research effort — it
is the deterministic output of the kernel inventory plus the profiling data.

---

## 3. Core Principles

Every kernel in LonghornAI must include all nine of the following artifacts.
This is a **gate**, not a guideline. A kernel is "not done" until all are
present.

1. **Mathematical specification** — the operation written in unambiguous
   mathematical form, including index ranges, dtypes, broadcasting rules,
   numerical-stability tricks (max-subtraction, Welford, Kahan, etc.), and
   tolerance conventions.
2. **Naive reference implementation** — a triple-nested-loop, obviously
   correct, FP32-only implementation. This is the *golden oracle*. It is
   never optimized. It is never modified for performance.
3. **Optimized implementation** — cache-aware, blocked, optionally
   multithreaded (OpenMP), still pure portable C++ in the early phases. May be
   followed by SIMD/vendor-intrinsic variants in later phases under build
   flags, never as the default.
4. **Unit tests** — kernel-vs-reference differential tests across a random
   shape suite, deterministic seeds, edge cases (empty, single-element, large
   magnitude, denormals, non-aligned strides).
5. **Numerical validation** — per-dtype tolerance bounds, max-ULP and
   relative-error reporting on a fixed validation set, stability checks under
   extreme inputs.
6. **Performance benchmarks** — latency, throughput, achieved GFLOPS, achieved
   bandwidth, distribution (not just mean), warm-up and steady-state phases,
   reported across the model-shape suite.
7. **Memory traffic analysis** — bytes read, bytes written, working set size,
   reuse distance, and arithmetic intensity (FLOPs per byte) at each problem
   size.
8. **Hardware mapping notes** — which silicon block(s) would execute this
   kernel: tensor unit, vector unit, reduction tree, softmax engine,
   normalization engine, attention engine, KV-cache controller, DMA engine.
9. **Accelerator implications** — required datapath widths, accumulator
   precision, on-chip SRAM, NoC bandwidth, and any non-obvious microarchitectural
   asks (e.g. "online-softmax requires a streaming exp/log unit and a fused
   max-rescale path").

A standardized `KERNEL.md` template in `docs/templates/` codifies this.

---

## 4. Repository Architecture

The repository scales from the existing minimal C++ kernel set to a multi-tier
project. The target layout:

```
LonghornAI/
  CMakeLists.txt
  PLAN.md                       this document
  README.md
  CONTRIBUTING.md
  docs/
    templates/
      KERNEL.md                 the 9-section kernel spec template
    architectures/              one .md per supported model family
    silicon/                    the Phase 8/9 architecture documents
    benchmarks/                 saved benchmark reports (CSV + analysis)
  src/
    core/
      dtypes.hpp                fp32, fp16, bf16, fp8 (e4m3/e5m2), int8/4/2,
                                 mxfp helpers; portable software conversion
      tensor.hpp/.cpp           owning tensor with allocator hooks
      tensor_view.hpp           non-owning shape/stride view (already exists)
      allocator.hpp/.cpp        arena, slab, page-aligned, NUMA-aware
      device.hpp                abstract device handle (CPU today; ext later)
      profiler.hpp/.cpp         per-kernel counters, traces, JSON export
      tracing.hpp/.cpp          chrome-trace-format event log
    kernels/
      gemm/                     dense, batched, grouped, quantized variants
      norm/                     LayerNorm, RMSNorm, GroupNorm
      activation/               GELU(erf/tanh), SiLU, ReLU, Tanh, Sigmoid,
                                 SwiGLU, GeGLU
      softmax/                  stable, online, fused-with-mask
      reduction/                sum, max, mean, argmax, top-k
      transform/                transpose, permute, reshape, concat, split,
                                 gather, scatter, embedding
      rope/                     rotary, NTK, YaRN, ALiBi, relative
      attention/                SDPA, masked, causal, MHA, MQA, GQA,
                                 sliding-window, chunked, FlashAttn,
                                 FlashDecoding, cross
      kvcache/                  contiguous, blocked, paged, prefix, quantized
      quant/                    int8/4/2, fp8, mxfp, awq, gptq, smoothquant,
                                 hqq, gguf-style; per-channel/group/token
      moe/                      router, dispatch, combine, grouped GEMM
      ssm/                      Mamba/Mamba-2 selective scan, RWKV, RetNet,
                                 Hyena, linear attention
      sampling/                 greedy, temperature, top-k, top-p, min-p,
                                 typical, mirostat, beam, speculative
    runtime/
      model.hpp/.cpp            model graph, weight binding, forward()
      loader/                   safetensors, GGUF, HF-format weight loaders
      tokenizer/                BPE, SentencePiece, tiktoken-compatible
      scheduler/                continuous/dynamic batching, request queue
      cache_manager.hpp/.cpp    KV cache lifetime, eviction, paging
      decode/                   draft, target, verifier (speculative)
    bench/
      bench_main.cpp            harness with warm-up, distribution reporting
      shapes/                   YAML shape suites per model family
    tools/
      validate_model.cpp        load checkpoint, run forward, compare to HF
      profile_model.cpp         per-layer / per-kernel profiling
      kernel_explorer.cpp       interactive shape-sweep tool
  tests/
    unit/                       per-kernel differential tests (GoogleTest)
    integration/                end-to-end forward-pass tests
    numerical/                  ULP/relative-error sweeps
    stress/                     long-context, large-batch, memory-pressure
  third_party/                  GoogleTest (FetchContent), nothing else heavy
```

The current `src/kernels/*.{hpp,cpp}` files become the seed of `kernels/` and
get reorganized into per-family subdirectories during Phase 0.

---

## 5. Build, Tooling, and Portability

- **Language:** C++17 baseline, optional C++20 modules behind a flag.
- **Build:** CMake ≥ 3.20, single-config and multi-config aware. C++ only —
  `LANGUAGES CXX`. No CUDA enabled in the root project. GPU/accelerator
  targets, when added, live under `experimental/` with their own
  `add_subdirectory` and a build flag (`-DLONGHORN_ENABLE_CUDA=ON`).
- **Compilers:** MSVC, GCC ≥ 9, Clang ≥ 11. Per-compiler warning flags.
- **Threading:** OpenMP optional (`-DLONGHORN_ENABLE_OPENMP=ON`). Single-thread
  must always pass.
- **SIMD:** No intrinsics in core kernels. Optional intrinsic variants
  (`-DLONGHORN_ENABLE_AVX2=ON`, `-DLONGHORN_ENABLE_NEON=ON`) live in
  `kernels/<family>/simd/` and must agree bit-for-bit with the portable path
  within tolerance.
- **Test framework:** GoogleTest via `FetchContent`. CTest as the runner.
- **Bench framework:** in-house, lightweight, JSON+CSV output, percentile
  reporting (P50/P90/P99), warm-up + steady-state phases.
- **CI:** GitHub Actions matrix over (Linux/macOS/Windows) × (GCC/Clang/MSVC)
  × (Debug/Release) × (OpenMP on/off). Test + benchmark smoke must pass.
- **Determinism:** every random tensor uses a seeded `std::mt19937_64`. No
  `Date.now`/`std::chrono`-derived randomness anywhere in tests.

---

## 6. Cross-Cutting Concerns

### 6.1 Numerics

- FP32 is the **source of truth** for all reference implementations.
- FP16, BF16, FP8 (E4M3 and E5M2), MXFP4/6/8, INT8/4/2 are **storage and
  reduced-precision compute** dtypes. Each has documented absolute and
  relative tolerance bounds in `docs/numerics.md`.
- Reductions (norms, softmax, dot products) accumulate in FP32 regardless of
  input dtype.
- Stability tricks must be commented at the line they appear (max-subtraction,
  Welford one-pass variance, Kahan summation, log-sum-exp identity, online
  softmax rescaling).

### 6.2 Memory model

- Tensors are described by `(dtype, shape, strides, data_ptr, device)`.
- Strides are explicit; no kernel assumes contiguity unless its contract says
  so. Contiguity is checked at entry and either reported, refused, or
  implicitly handled by a copy-in helper.
- Allocators are pluggable: arena (bump), slab (per-size-class), page-aligned
  (for paged KV cache), NUMA-aware. Default for tests is the arena.

### 6.3 Profiling and tracing

- Every kernel is wrapped in a `LH_KERNEL_SCOPE("name", shape...)` macro that,
  when enabled, records start/end timestamps, byte counts, FLOP counts, and
  emits a Chrome-trace-format JSON.
- `tools/profile_model.cpp` runs a forward pass and produces
  `(kernel, latency, GFLOPS, GB/s, arithmetic_intensity)` per call,
  aggregated across an inference batch.

---

# Phase 0 — Infrastructure

**Goal:** a building, testable, profileable skeleton that every subsequent
phase plugs into without modification.

## 0.1 Repository scaffold

- Reorganize `src/kernels/*.{hpp,cpp}` into the per-family directory layout.
- Add `src/core/` with `dtypes.hpp`, `tensor.hpp`, `tensor_view.hpp`,
  `allocator.hpp`, `device.hpp`, `profiler.hpp`, `tracing.hpp`.
- Add `docs/templates/KERNEL.md` — the 9-section template.
- Add `CONTRIBUTING.md` enumerating the kernel-completion gate.

## 0.2 Tensor abstraction

- `Tensor`: owning, RAII, allocator-backed; supports view-of, slice, contiguous
  copy.
- `TensorView`: non-owning; data pointer + shape + strides + dtype tag.
- `Shape`: small-vector-of-int64 with rank ≤ 8; supports broadcasting,
  squeeze, unsqueeze, permute.
- Index helpers: `linear_index(strides, multi_index)`, iterator-of-indices.

## 0.3 Memory allocators

- `ArenaAllocator` — bump pointer, used for tests and per-forward scratch.
- `SlabAllocator` — power-of-two size classes; for KV cache blocks.
- `PageAllocator` — page-aligned (4 KiB / 2 MiB), for paged-attention blocks.
- `NumaAllocator` — pin to a NUMA node when present.
- All implement an `IAllocator` interface with `alloc(size, align)` /
  `free(ptr)` / `reset()`.

## 0.4 Benchmark and test framework

- `tests/test_util.hpp`: `expect_close(a, b, dtype)`, random tensor generators,
  shape iterators, ULP comparison, golden-file comparisons.
- `bench/bench_main.cpp`: warm-up N iters, measure M iters, report
  P50/P90/P99/mean/std, FLOPS, GB/s, arithmetic intensity. Writes
  `bench/results/<date>.json`.
- Shape suites under `bench/shapes/`: `llama-7b.yaml`, `llama-70b.yaml`,
  `mixtral-8x7b.yaml`, `qwen-2-72b.yaml`, etc. — drives benchmarks and
  validation alike.

## 0.5 Numerical validation framework

- `tests/numerical/ulp_sweep.cpp`: per-kernel sweep over (dtype, shape, value
  distribution) producing max ULP, mean ULP, and a histogram per output bin.
- Tolerance policy: FP32 rel ≤ 1e-5 abs ≤ 1e-7; FP16 rel ≤ 1e-3 abs ≤ 1e-4;
  BF16 rel ≤ 5e-3 abs ≤ 5e-4; INT8 exact within quantization scheme.

## 0.6 Reference model runner

- `runtime/model.{hpp,cpp}`: declarative graph of `Op` nodes, each pointing to
  a kernel. `forward(input, kv_cache)` returns logits.
- Initial implementation supports a hand-coded Llama-2-7B graph as the
  end-to-end test of Phase 1 + 2 + 3.
- A `validate_model` CLI loads HF weights, runs forward, compares last-hidden
  to a reference dump (produced offline via PyTorch, checked into
  `tests/integration/golden/`).

## 0.7 Weight loading and model import

- Loaders for: `safetensors`, `GGUF`, raw HuggingFace `pytorch_model.bin` (via
  pickle-free safetensors-only path; we do not parse pickle).
- Mmap-based zero-copy where possible.
- `loader/transform.{hpp,cpp}`: post-load weight transforms (transpose for
  gemm conventions, fuse `qkv` into a single matrix, pre-permute MoE
  experts).

## 0.8 Profiling and tracing tools

- `profiler.{hpp,cpp}`: process-wide registry of kernel scopes; aggregates
  count, total time, total bytes, total FLOPs.
- `tracing.{hpp,cpp}`: emits Chrome-trace JSON, viewable in `chrome://tracing`
  or Perfetto.
- `tools/profile_model.cpp`: runs a forward pass under the profiler, prints a
  ranked table by total time and by total bytes.

## 0.9 Performance counters

- Optional Linux `perf_event_open` integration behind
  `-DLONGHORN_ENABLE_PERF=ON`: cycles, instructions, L1/L2/LLC misses,
  branches.
- Reports per-kernel IPC, miss rates, and memory traffic.

## 0.10 Acceptance criteria — Phase 0

- [ ] Build is green on Linux/macOS/Windows × GCC/Clang/MSVC × Debug/Release.
- [ ] At least one trivial kernel routed end-to-end through `Tensor`,
      `Allocator`, `Profiler`, `Tracer`.
- [ ] Numerical-validation harness produces a ULP report on the existing
      activation kernel.
- [ ] Benchmark harness reports P50/P90/P99 on the existing GEMM kernel.
- [ ] Llama-2-7B weight loader successfully mmaps a safetensors checkpoint and
      enumerates parameters.

---

# Phase 1 — Foundational Math Kernels

**Goal:** every primitive that a transformer block multiplies, adds,
normalizes, or activates with. The dense-compute bedrock.

## 1.1 GEMM family

| Kernel | Math | Hardware lens |
|---|---|---|
| Dense GEMM | `C = α·A·B + β·C` | Compute-bound for large M/N/K. Systolic-friendly. Demands FP32 accumulator regardless of input dtype. |
| Batched GEMM | per-batch `C_b = α·A_b·B_b + β·C_b`, uniform shape | Same as dense; batch dim hides pipeline bubbles. |
| Grouped GEMM | per-group `C_g = α·A_g·B_g + β·C_g`, ragged shape | Foundation for MoE; needs a dispatch unit that can issue many small GEMMs without per-issue overhead. |
| MatVec / GEMV | `y = α·A·x + β·y` | Memory-bound. Decode-time hot path. Demands a streaming weight-fetch + reduction tree. |
| Dot products | `c = sum_i a_i b_i` | Memory-bound at small sizes; building block of MatVec, attention scores. |

**Implementation roadmap:** naive triple-loop → 4-level blocking (register /
L1 / L2 / DRAM) → optional packing → optional micro-kernel SIMD path under
flag.

**Optimization roadmap:** identify the right `MR×NR` register tile per
target; pack `B` panels for unit-stride access; reuse `A` across `NR`; for
MatVec, prefer a streaming layout that overlaps weight load and reduction.

**Hardware implications:** a tensor unit with FP16/BF16/FP8/INT8 inputs and
FP32 accumulators; a separate vector path for MatVec (decode), since reusing
a square systolic array for skinny matrices wastes MACs.

## 1.2 Reductions and broadcasts

- `sum`, `max`, `min`, `mean`, `argmax`, axis reductions.
- `top-k` (heap-based + sorted-output variants).
- Broadcasting rules: NumPy-style; explicit broadcast helper that yields
  (shape, strides) without copying when possible.

**Hardware implications:** reduction trees with FP32 accumulation; a max-
subtraction primitive that fuses with downstream softmax.

## 1.3 Elementwise

- Add, sub, mul, div, neg, abs, exp, log, sqrt, rsqrt, pow, clamp.
- Fused variants used by other kernels (`a*b + c`, `(x - μ) * rsqrt(σ² + ε)`).

**Hardware implications:** vector unit with reciprocal/rsqrt approximations;
exp/log polynomial-approx pipelines (used by softmax, normalization, GELU).

## 1.4 Tensor transforms

- `transpose`, `permute`, `reshape` (zero-copy when stride-compatible),
  `concat`, `split`, `gather`, `scatter`, `embedding lookup`.
- All implementations must support non-contiguous strides without a forced
  copy.

**Hardware implications:** these are **DMA work**. A capable DMA engine with
strided + scatter-gather + transposed-tile modes eliminates the need to
execute most of these on the compute fabric.

## 1.5 Normalization

- **LayerNorm:** `(x - μ) / sqrt(σ² + ε) · γ + β`. Welford one-pass for
  stability.
- **RMSNorm:** `x · rsqrt(mean(x²) + ε) · γ`. Llama / Qwen / Mistral.
- **GroupNorm:** rare in LLMs; included for vision/multimodal extensions.

**Hardware implications:** a normalization engine with FP32 reduction lanes,
fused rsqrt, and per-row broadcast of (γ, β). Two passes (mean → variance)
are eliminated by Welford in one.

## 1.6 Softmax

- Stable per-row softmax: `exp(x - max(x)) / sum(...)`.
- **Online softmax** (running max + running sum) for streaming and FlashAttn.
- Fused `mask + scale + softmax` variant (eliminates a pass).
- `log_softmax` for sampling and log-likelihood.

**Hardware implications:** a softmax engine with streaming exp + max-rescale
+ reciprocal in one pipeline. The online formulation is the silicon-friendly
form because it never materializes the score matrix.

## 1.7 Activations

| Kernel | Math |
|---|---|
| GELU (erf) | `½ x (1 + erf(x/√2))` |
| GELU (tanh) | `½ x (1 + tanh(√(2/π)(x + 0.044715 x³)))` |
| SiLU / Swish | `x · σ(x)` |
| ReLU | `max(0, x)` |
| Sigmoid | `1 / (1 + e⁻ˣ)` |
| Tanh | `(eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)` |
| SwiGLU | `SiLU(a) ⊙ b` (gated MLP) |
| GeGLU | `GELU(a) ⊙ b` (gated MLP) |

**Hardware implications:** all are pointwise, fully vectorizable, fuseable
into the preceding GEMM's epilogue. SwiGLU/GeGLU motivate a **fused MatMul +
gate + activation** path (the Llama MLP block) since gate + up + activation
+ down is the single hottest fused pattern in the model.

## 1.8 Phase 1 — Acceptance criteria

- [ ] Every kernel has the 9 artifacts (math, naive, optimized, tests,
      numerics, bench, traffic, hw-mapping, accel-implications) committed.
- [ ] Optimized GEMM ≥ 30% of theoretical peak FP32 on a modern x86 core in
      Release on the model-shape suite.
- [ ] Optimized RMSNorm/Softmax ≥ 70% of memory bandwidth on long rows.
- [ ] All Phase-1 kernels match the reference within per-dtype tolerance
      across the random + edge-case suites.

---

# Phase 2 — Transformer Attention Stack

**Goal:** every attention variant a modern decoder might use, plus all
positional encoding schemes, all built on Phase-1 primitives.

## 2.1 SDPA family

- **Scaled Dot-Product Attention:** `softmax(QKᵀ/√d + M) V`.
- **Masked / causal:** `M_ij = -∞ for j > i`.
- **MHA:** independent K/V per query head.
- **MQA:** one K/V head shared across all query heads.
- **GQA:** `H_q / H_kv` query heads per K/V head — used by Llama-3 / Qwen-2.
- **Cross-attention:** distinct Q stream from a different sequence (encoder/
  decoder, retrieval-augmented).
- **Sliding-window attention:** `M_ij = -∞ for |i - j| > W`. Mistral-style.
- **Chunked attention:** local within fixed-size chunks plus optional global
  tokens.

## 2.2 Flash family

- **FlashAttention** (v1-style): tiled, online-softmax, no materialized score
  matrix. Per key block, maintain `(m, ℓ)` and rescale `O` on each new max.
- **FlashAttention-2**: improved work partitioning (iterate over Q outer, K
  inner per query head); split-K parallelism for short-Q long-K (decode).
- **FlashDecoding**: split the K/V dimension across compute units, then
  reduce. Critical for low-batch decode where the K/V dimension dominates.
- **Speculative-decoding attention support**: contiguous-Q attention over a
  tree-shaped K/V mask (Medusa, Eagle).

## 2.3 Long-context attention

- Sliding-window with bias toward recent tokens.
- Chunked attention with cross-chunk summaries.
- Linear attention forms (kernel-feature trick); foundation for Phase 7's
  RetNet and Hyena.

## 2.4 Positional encodings

- **RoPE** (interleaved and half-rotation layouts).
- **NTK-aware RoPE scaling** — adjusts the base θ to extend context.
- **YaRN** — frequency-band-wise scaling.
- **ALiBi** — additive linear bias, no learned positional embedding.
- **Relative position encodings** (T5-style) for completeness.

## 2.5 Composable attention building blocks

- `attention_score(Q, K, scale, mask)` → scores
- `attention_output(scores, V)` → output
- `attention_full(Q, K, V, mask, ...)` — ties them together
- Each piece is independently testable and benchmarkable; the silicon
  partitioning naturally falls along these boundaries.

## 2.6 Hardware implications

- **Attention is the highest-arithmetic-intensity hotspot at long context**
  in prefill, and the highest-bandwidth-pressure operation in decode. It
  motivates a dedicated **Attention Engine** with:
  - Streaming Q tile loader.
  - K/V tile streamer with online-softmax accumulator.
  - Fused score-scaling + masking + softmax + value-multiply pipeline.
  - FP32 accumulators for the running `(m, ℓ, O)` state.
- **RoPE / ALiBi** are pre-attention pointwise transforms, fuseable into the
  Q/K projection epilogue; do not need a separate engine.
- **GQA / MQA** reduce K/V bandwidth by `H_q/H_kv`× — the silicon must
  exploit this by *not* fetching duplicate K/V. The KV cache controller
  must be GQA-aware.

## 2.7 Phase 2 — Acceptance criteria

- [ ] FlashAttention matches SDPA reference within tolerance on causal,
      non-causal, MHA/MQA/GQA, and sliding-window suites.
- [ ] FlashDecoding correctness on (B=1, S_kv up to 32k) decode shapes.
- [ ] All RoPE variants reproduce HF reference outputs on a Llama-2 layer.
- [ ] Cross-attention validated against an encoder-decoder reference (T5).

---

# Phase 3 — KV Cache System

**Goal:** the storage, scheduling, and memory subsystem that makes
multi-request decode feasible. This is the deepest phase in the plan because
the KV cache is **the** dominant memory pressure in production serving and
the single largest driver of accelerator memory architecture.

## 3.1 Background — why this matters

For a 70B-class Llama, the per-token KV cache is on the order of hundreds of
KB; a 4 K-token request consumes hundreds of MB; a thousand concurrent
requests consume hundreds of GB. The bandwidth required to read this cache
on every decode step is what dictates whether a serving GPU/accelerator is
usable at all. **The KV cache, not the weights, is what makes inference hard
at scale.**

A serving stack that does not solve this problem cannot ship. A silicon
design that does not solve this problem cannot exist.

## 3.2 Cache layout variants

- **Contiguous cache** — one big `[B, H_kv, S_max, d]` tensor per layer.
  Simple, fast, but allocates `S_max` even for short requests, wasting
  memory. Forces a fixed maximum sequence length per request.
- **Blocked cache** — chunked into fixed-size blocks of (e.g.) 16 tokens.
  Same `S_max` problem unless paired with paging.
- **Paged cache** (vLLM / PagedAttention) — physical KV blocks of K tokens
  each, indirected through a per-request **block table**. Eliminates
  fragmentation; lets multiple requests share blocks (prefix caching).
  Decoupled physical/logical layout is the silicon-relevant insight.
- **Virtualized cache** — paged plus an OS-style page table; supports
  swap-in/swap-out to host RAM or NVMe under memory pressure.

## 3.3 Prefix caching and shared-prefix reuse

- **Prefix cache:** persist KV blocks for repeated system prompts and
  document prefixes; reuse across requests via the block table.
- **Radix tree of prefixes** (SGLang-style) to maximize sharing; LRU eviction
  at the tree level.
- Bit-exact reuse — no recomputation when the prefix matches.

## 3.4 KV cache compression and quantization

- **INT8 KV cache** — per-channel scale; halves bandwidth at small numerical
  cost.
- **FP8 KV cache** (E4M3 / E5M2) — preserves dynamic range with one byte.
- **INT4 KV cache** — group-wise scale; quartered bandwidth, used by recent
  Llama deployments.
- **Sparse / pruned KV** (H2O, Heavy-Hitters) — drop low-attention tokens
  beyond a recency window.
- **MLA (DeepSeek)** — multi-head latent attention compresses K/V into a
  joint low-rank latent; bandwidth wins of an order of magnitude.

## 3.5 Multi-request KV cache and scheduling

- **Continuous batching** — at every decode step, evict completed sequences
  and admit new ones; never wait for a full batch.
- **Dynamic batching** — group prefill and decode together with a
  micro-batching scheduler.
- **Request scheduling** — priority queues, fair queuing, deadline-aware
  scheduling, anti-starvation for long requests.
- **Decode scheduling** — a dispatcher that walks the block tables and emits
  one `attention(decode)` per active request per step, batched into a single
  kernel where possible.
- **Cache eviction** — LRU on prefix tree; FIFO on per-request blocks once
  request completes.
- **Cache compaction** — defragment block tables when fragmentation crosses
  a threshold.
- **Cache migration** — move blocks between memory tiers (HBM ↔ host ↔ NVMe)
  under pressure, asynchronously via DMA.

## 3.6 vLLM, TensorRT-LLM, and SGLang lessons

- **vLLM** — paged attention; physical/logical decoupling; copy-on-write
  for branched decode (beam, parallel sampling); near-zero fragmentation.
- **TensorRT-LLM** — in-flight batching; per-batch fused attention kernels;
  XQA-style decoder-specific fast paths; FP8 KV across the board.
- **SGLang** — RadixAttention prefix cache; structured generation co-design
  with the KV layout; speculative-decoding-aware cache.

LonghornAI must absorb each of these techniques as first-class kernels and
runtime components, not as third-party integrations.

## 3.7 Memory bandwidth analysis

For each layout, document at the per-token, per-request, and per-batch
levels:

- Bytes **read** per decode step (K+V over the cache + new K/V append).
- Bytes **written** per decode step (new K/V append only).
- Reuse distance — how far apart two reads of the same KV block are in
  decode steps.
- Effective bandwidth — what fraction of HBM bandwidth is actually used.

This data is the input to Phase 8's silicon classification of the KV-cache
controller.

## 3.8 Long-context support

- Up to 1M-token context targeted (Llama-3.1, Qwen-2.5, GPT-4-class).
- KV must be paged, quantized, and possibly sparse/pruned to fit.
- Attention must be FlashDecoding-tiled with split-K parallelism.
- Prefill must be FlashAttention-2 with chunked Q.

## 3.9 Silicon implications — KV cache controller

The KV cache motivates a **dedicated KV controller block** with:

- Block-table walker (logical-to-physical).
- Multi-stream prefetcher (predicts next-needed blocks).
- On-the-fly dequant (INT4/INT8/FP8 → FP16/BF16) on the read path.
- Append path (new K/V → next block) with quantization on write.
- Eviction / migration engine driven by DMA.
- Coherence with the attention engine — single-cycle handoff of (K, V, mask)
  tiles.

## 3.10 Phase 3 — Acceptance criteria

- [ ] Paged-attention kernel matches SDPA reference for all layouts.
- [ ] Continuous-batching scheduler sustains ≥ 80% kernel occupancy on a
      synthetic mixed-length workload.
- [ ] Prefix cache delivers ≥ 95% of theoretical reuse on a system-prompt
      benchmark.
- [ ] INT8/FP8 KV cache within documented numerical tolerance vs. FP16 KV.
- [ ] 1M-context Llama-3.1 forward pass runs end-to-end without OOM on a
      target memory footprint.

---

# Phase 4 — Quantization

**Goal:** every quantization scheme a modern inference stack uses, with
hardware-realistic datapaths. This is where silicon datapath width is set.

## 4.1 Dtype inventory

- **FP32** — reference; never used for production weights/activations.
- **FP16** — IEEE-754 binary16; legacy production.
- **BF16** — same exponent as FP32, 7-bit mantissa; production default.
- **FP8** — `E4M3` (weights, activations) and `E5M2` (gradients/dynamic
  range). Per-tensor or per-channel scale.
- **MXFP** (Microscaling) — block-shared exponent over 32-element groups,
  6/4-bit element mantissas. **The strongest current candidate for
  hardware-friendly low-precision.**
- **INT8** — symmetric/asymmetric, per-tensor/per-channel/per-token scale.
- **INT4** — per-group (group size 32/64/128), zero-point optional.
- **INT2** — extreme; matters for KV and embeddings, not for compute.

## 4.2 Quantization schemes

- **Per-tensor** — one scale per tensor; fast but low fidelity.
- **Per-channel** — one scale per output channel; standard for weights.
- **Per-group** — one scale per group of K (e.g. 32, 64, 128) elements
  along an axis; standard for INT4 weights.
- **Per-token** — one scale per token (activation) or per row.
- **Block / MX** — per-block (e.g. 32 elements) shared exponent.

## 4.3 Algorithms

- **AWQ** — Activation-aware Weight Quantization; per-channel scaling
  pre-quant, INT4 weights.
- **GPTQ** — Hessian-aware error-minimizing INT4 weight quantization.
- **SmoothQuant** — migrate quantization difficulty from activations to
  weights via channel scales.
- **HQQ** — Half-Quadratic Quantization; weight-only, fast calibration-free.
- **GGUF-style** — k-quants (Q2/Q3/Q4/Q5/Q6/Q8 with sub-block scale +
  super-block scale); the de-facto edge format.

## 4.4 Quantized kernels

- **W8A8 GEMM** — INT8 weights × INT8 activations, INT32 accumulate, FP32
  rescale.
- **W4A16 GEMM** — INT4 weights × FP16 activations, on-the-fly dequant in
  the inner loop, FP32 accumulate.
- **W4A8 GEMM** — INT4 weights × INT8 activations.
- **FP8 GEMM** — E4M3 × E4M3, FP32 accumulate.
- **MXFP GEMM** — block-exponent decoded on-the-fly.
- **KV-cache quant** — INT8/FP8/INT4 K and V; on-the-fly dequant on read.

## 4.5 Hardware datapath implications

The quantization phase **sets the silicon's datapath width**. Concretely:

- Tensor unit must support FP16/BF16/FP8/INT8/INT4 inputs, FP32/INT32
  accumulators, with **fused dequant on the input feeders** so the MAC array
  always sees a uniform compute dtype.
- MAC unit bit-widths: a `FP32 accumulator + FP16 mantissa multiplier` is
  the workhorse; INT8 multipliers + INT32 accumulators run at 2× the
  throughput; INT4 multipliers fuse two pairs per cycle.
- Accumulator precision is **never** quantized — FP32 accumulation is a hard
  silicon requirement to preserve numerical stability.
- A **rescale unit** at the GEMM epilogue applies per-channel/group/token
  scales before downstream ops.

## 4.6 Phase 4 — Acceptance criteria

- [ ] W4A16 GEMM matches FP16 GEMM within documented tolerance on the
      Llama-3-70B shape suite, with > 3× memory bandwidth reduction.
- [ ] FP8 (E4M3) GEMM end-to-end forward on Llama-3-8B with < 0.1 perplexity
      delta vs. BF16.
- [ ] AWQ + GPTQ + GGUF Q4_K_M loaders ingest reference checkpoints and
      validate forward equivalence.
- [ ] INT8/FP8 KV cache within tolerance on a long-context perplexity test.

---

# Phase 5 — Mixture of Experts

**Goal:** the kernel set required by Mixtral, DeepSeek-MoE, and Qwen-MoE.

## 5.1 MoE building blocks

- **Router** — small dense (typically a `Linear(d, n_experts)` + softmax).
- **Top-k routing** — for each token, pick the top-k expert indices and
  their gate weights (Mixtral: top-2; DeepSeek: top-2 with shared experts).
- **Dispatch** — gather token activations into per-expert contiguous
  buffers (token-major → expert-major reshuffle).
- **Expert MLP** — per-expert SwiGLU/GeGLU MLP; runs as a **grouped GEMM**
  across all experts in parallel.
- **Combine** — scatter expert outputs back to token positions, weighted by
  the router gate.

## 5.2 MoE variants

- **Mixtral-style** — 8 experts, top-2; standard SwiGLU MLP.
- **DeepSeek-style** — fine-grained experts (64+), shared experts always
  active, expert-load-balancing routing tricks.
- **Qwen-MoE** — similar fine-grained scheme.

## 5.3 Sparse activation kernels

- Token-to-expert dispatch is a **scatter**.
- Expert-to-token combine is a **gather** with weighted sum.
- These are *the* MoE-specific kernels; they require a high-throughput
  permutation / shuffle path on silicon.

## 5.4 Memory and compute implications

- MoE explodes weight memory (`n_experts ×` MLP weights) but keeps active
  compute roughly constant per token (`top_k / n_experts ×` of dense MLP).
- Implication: MoE is **dramatically more memory-bound** than dense models.
  The bandwidth win from quantization is more valuable here than anywhere
  else.
- Grouped GEMM is the central kernel; it must amortize dispatch overhead by
  issuing many ragged small GEMMs without per-issue cost.

## 5.5 Silicon implications — MoE

- **Expert dispatch unit** — a permutation/scatter engine that shuffles
  tokens into expert lanes.
- **Grouped GEMM** native dispatch — the tensor unit must accept a list of
  `(M_g, N_g, K_g)` tuples and run them back-to-back without host
  intervention.
- **Cross-chip / cross-die expert parallelism** — for many-expert models,
  experts shard across compute tiles; a low-latency interconnect with
  all-to-all support is required.
- **Routing fast-path** — the router itself is a small dense GEMM + softmax
  + top-k; a small dedicated lane can pipeline it with the previous layer's
  attention output.

## 5.6 Phase 5 — Acceptance criteria

- [ ] Mixtral-8×7B end-to-end forward bit-exact (within FP tolerance) vs.
      reference HF implementation.
- [ ] DeepSeek-MoE end-to-end forward.
- [ ] Grouped GEMM hits ≥ 70% of dense GEMM throughput when expert load is
      balanced.
- [ ] Dispatch + combine fused into a single shuffle pass.

---

# Phase 6 — Modern Inference Optimizations

**Goal:** the runtime techniques that turn correct kernels into a serving
system.

## 6.1 Batching strategies

- **Continuous batching** — admit/evict per step.
- **Dynamic batching** — variable batch sizes based on queue depth.
- **Micro-batching** — split a large batch into pipeline-overlapping micro-
  batches to hide all-reduce latency in tensor-parallel deployments.
- **Prefill / decode separation** — prefill is compute-bound, decode is
  memory-bound; run them on separate streams (or even separate devices in
  disaggregated serving).

## 6.2 Caching

- **Prefix caching** — Phase 3.
- **Prompt caching** — at the system-prompt level; multi-tenant; LRU.
- **Embedding cache** — for repeated tokens in the embedding lookup.

## 6.3 Speculative decoding

- **Draft model** — a small model proposes the next K tokens.
- **Target model** — the big model verifies them in a single forward pass
  over K candidate positions.
- **Acceptance** — accept the longest matching prefix; resample on
  rejection.
- **Medusa** — multiple decoding heads in the same model predict
  `t+1, t+2, t+3, ...` simultaneously; tree-shaped verification.
- **Eagle** — feature-level draft model; higher acceptance rate than Medusa.
- **Lookahead decoding** — Jacobi-iteration-style parallel decoding without
  a draft model.
- **Tree decoding** — verify a tree of candidate continuations in one
  forward pass; arbitrary mask shape required.
- **Parallel decoding** — variants on top of lookahead/Medusa.

## 6.4 Sampling

- **Greedy / argmax**.
- **Temperature** scaling.
- **Top-k**, **Top-p (nucleus)**, **Min-p**, **Typical**.
- **Mirostat** — surprise-controlled sampling.
- **Beam search** — for completeness; rarely used in modern LLM serving.
- **Speculative-decoding sampler** — accept/reject under correct probability
  preservation.

## 6.5 Streaming and latency

- Token streaming — emit tokens as they decode, not at end of sequence.
- TTFT (time-to-first-token), ITL (inter-token-latency) — the two metrics
  that matter to users.

## 6.6 Scheduler design

- **Request queue** — priority + fair-queuing; SLO-aware.
- **Step scheduler** — at each decode step, pick the next batch composition
  to maximize throughput subject to ITL SLO.
- **Admission control** — refuse / queue when KV memory is exhausted.

## 6.7 Phase 6 — Acceptance criteria

- [ ] Continuous batching achieves ≥ 80% kernel occupancy on a 1k-request
      synthetic workload.
- [ ] Speculative decoding achieves ≥ 2× throughput on a target model with
      a properly-sized draft model.
- [ ] Medusa / Eagle decoding implementations match published acceptance
      rates within 5%.
- [ ] All sampling strategies validated against numerical golden traces.

---

# Phase 7 — Alternative LLM Architectures

**Goal:** ensure the kernel inventory covers every viable post-Transformer
architecture so the silicon does not over-specialize on attention.

## 7.1 RWKV (v4–v7)

- Recurrent linear-attention reformulation; per-token state update.
- Required kernels: time-mix (token shift), channel-mix (gated MLP), WKV
  recurrence, layer norm.
- Hardware view: streaming-friendly, very low arithmetic intensity per
  token, dominated by small MatVecs.

## 7.2 Mamba / Mamba-2

- **Selective State-Space Model** — `h_t = A_t · h_{t-1} + B_t · x_t`,
  `y_t = C_t · h_t + D · x_t`, where `A, B, C` are *input-dependent*.
- The hot kernel is the **selective scan** — a parallel-prefix scan over the
  sequence with per-token-varying `A, B, C`.
- Mamba-2 reformulates the scan as a structured matrix multiply (SSD form),
  bringing it onto the same tensor-unit datapath as attention.
- Required kernels: input-dependent scan, structured matmul (SSD), 1D conv,
  RMSNorm, gated MLP.

## 7.3 RetNet

- Retention mechanism — parallel, recurrent, and chunk-wise formulations of
  the same operation.
- Required kernels: retention (linear-attention-like), gated MLP, group
  norm.

## 7.4 Hyena

- Long convolution with implicit (FFT-parameterized) filters.
- Required kernels: 1D convolution (long), FFT, gating.

## 7.5 Linear attention

- `softmax(QKᵀ)V` replaced by `φ(Q)(φ(K)ᵀV)` with a feature map `φ`.
- Required kernels: feature-map evaluation, outer product accumulation,
  cumulative sum.

## 7.6 Hybrid Transformer-SSM

- Jamba-style — interleave Mamba and Transformer blocks.
- Same kernel inventory as the union of attention + SSM; the *runtime* must
  switch between Phase 2 and Phase 7 per layer.

## 7.7 Hardware implications and comparison

Architectures are classified on three axes:

| Architecture | Compute | Memory | State |
|---|---|---|---|
| Transformer | high | KV-cache-bound | KV cache |
| Mamba / SSD | medium | weight-bound | small recurrent state |
| RWKV | low | weight-bound | small recurrent state |
| RetNet | medium | mixed | medium state |

Recurrent architectures' silicon implication is a **compact recurrent state
buffer** (much smaller than KV cache) and a **scan engine** that performs
parallel-prefix in hardware. The good news: Mamba-2's SSD reformulation
brings most of the heavy compute onto the same tensor unit as attention.

## 7.8 Phase 7 — Acceptance criteria

- [ ] Mamba-2 and RWKV-v6 forward passes match reference implementations.
- [ ] Hybrid Jamba-style model end-to-end forward.
- [ ] SSD (structured state space) compute path shares ≥ 80% of attention's
      tensor-unit code.

---

# Phase 8 — Silicon-Oriented Kernel Analysis

**Goal:** a complete, per-kernel classification table that becomes the input
to Phase 9. This phase produces no new kernels — it produces the
**architecture document**.

## 8.1 Classification axes

For every kernel in the repository, classify on:

- **Compute-bound vs. memory-bound** — at the model-shape suite, what is the
  measured arithmetic intensity vs. the roofline?
- **Bandwidth-limited vs. latency-limited** — is the bottleneck DRAM
  bandwidth or DRAM latency (small-tensor ops)?
- **Vectorizable** — does it map cleanly onto SIMD lanes?
- **Systolic-friendly** — does it map onto an MK × KN array with constant
  weight reuse?
- **Streaming-friendly** — can it be pipelined as a producer-consumer with
  minimal state?
- **Cache-sensitive** — does its performance depend on a specific level of
  cache fitting a tile?

## 8.2 Accelerator block candidates

For every kernel, identify the accelerator block(s) that would execute it:

- **Tensor units** — dense + grouped GEMM, attention score/output, FFN.
- **Vector units** — elementwise, activations, RoPE, quant rescale.
- **Reduction engines** — softmax, norm, top-k, dot products.
- **Softmax engines** — fused max-subtract + exp + sum + reciprocal.
- **Normalization engines** — fused Welford/mean-of-square + rsqrt + scale.
- **Attention engines** — fused QKᵀ + scale + mask + online-softmax + V.
- **KV cache controllers** — block-table walk, dequant, prefetch, append.
- **Scan engines** — Mamba-2 SSD, parallel prefix, RWKV WKV.
- **Permutation engines** — MoE dispatch/combine, transpose, gather/scatter.
- **DMA engines** — strided, scatter-gather, transposed-tile, descriptor-
  driven.

## 8.3 Memory hierarchy requirements

- **Register file** — per-lane working tiles (e.g. 16×16 FP16 per MAC tile).
- **L0 / line buffers** — per-engine streaming buffers (KV row, Q row).
- **L1 SRAM** — per-tile scratchpad sized for FlashAttention's online
  state (`(m, ℓ, O)` per query row × tile width).
- **L2 SRAM** — shared across tiles; sized to hold a layer's working set
  (a Q tile + multiple K/V tiles + accumulators).
- **HBM / off-chip DRAM** — KV cache, weights, embedding table.

Per-kernel SRAM working-set sizing is recorded in Phase 8 tables.

## 8.4 NoC and DMA requirements

- Bandwidth and latency budgets per kernel.
- Multicast support for weight broadcast.
- All-to-all support for MoE expert parallelism and tensor-parallel
  collectives.
- Coherence model — explicit, software-managed; no full cache coherence.

## 8.5 Per-kernel silicon dossier

Every kernel ends Phase 8 with a `docs/silicon/kernels/<name>.md` containing:

```
- Roofline position (AI vs. peak BW, peak FLOPS)
- Measured CPU performance on shape suite
- Memory traffic breakdown (read / write / reuse)
- Engine assignment
- SRAM working set size
- DMA pattern
- Datapath dtype requirements
- Open silicon questions
```

## 8.6 Phase 8 — Acceptance criteria

- [ ] Every kernel in the repo has a `docs/silicon/kernels/<name>.md`.
- [ ] A summary roofline plot is produced for the Llama-3-70B and
      Mixtral-8×7B shape suites.
- [ ] A consolidated table maps `kernel → engine → SRAM → DMA → dtype`.

---

# Phase 9 — Longhorn Accelerator Architecture

**Goal:** synthesize Phases 0–8 into a candidate Longhorn accelerator. This
is the **first** phase that produces hardware-facing artifacts.

## 9.1 Candidate compute organization

- **Tensor Tile** — the unit of replication. Each contains:
  - One **Tensor Unit** (systolic array, FP16/BF16/FP8/INT8/INT4 inputs,
    FP32/INT32 accumulators).
  - One **Vector Unit** (per-lane elementwise + activation pipelines).
  - One **Reduction Engine** (softmax/norm/top-k).
  - L1 SRAM scratchpad (sized by Phase 8).
  - L0 register file.
- **Attention Engine** — shared across N tiles; consumes Q/K/V tiles, runs
  online-softmax FlashAttention, emits output tiles.
- **KV Cache Controller** — block-table walker, prefetcher, on-the-fly
  dequant; sits between HBM and the attention engine.
- **Permutation / MoE Engine** — token-to-expert dispatch and combine.
- **Scan Engine** — Mamba-2 SSD and RWKV parallel-prefix.
- **DMA Engines** — multiple, descriptor-driven, supporting strided +
  scatter-gather + transposed.
- **Scheduler** — hardware request scheduler that issues per-step decode
  micro-programs across tiles.

## 9.2 Memory subsystem

- **Per-tile L1 SRAM** sized for one FlashAttention tile + one GEMM tile.
- **Shared L2 SRAM** sized for a layer's working set.
- **HBM** (or LPDDR for edge profiles) for weights and KV cache.
- **No coherence**; all transfers are software-orchestrated DMA.

## 9.3 Instruction format and execution model

- **Coarse-grained ISA** — one instruction issues a kernel-tile of work
  (e.g. a `GEMM_TILE`, a `FLASH_ATTENTION_TILE`, a `RMSNORM_ROW`).
- **Statically scheduled at compile time** by the LonghornAI compiler;
  hardware does no out-of-order issue.
- **VLIW-style bundles** for per-cycle dispatch across (Tensor, Vector,
  Reduction, DMA) units within a tile.
- **Predication and small-loop counters** for sequence-length variability.

## 9.4 Edge profile vs. server profile

- **Edge profile** — small accelerator, LPDDR memory, single-tile, focus on
  power-per-token. Targets 7B–13B dense models with 4-bit weights.
- **Server profile** — multi-tile, HBM, attention engines per tile, focus
  on tokens-per-second-per-watt at 70B+ scale.
- Both share the same ISA and the same kernel implementations; only
  resource counts and memory tiers differ.

## 9.5 Roadmap to silicon

- **Stage A — Cycle-accurate simulator (year 1):**
  - Model every block in C++ with cycle counts.
  - Run the full LonghornAI kernel suite through it.
  - Validate kernel-by-kernel against the CPU references bit-exactly.
  - Roofline-validate against the Phase 8 predictions.
- **Stage B — RTL prototype on FPGA (year 1.5–2):**
  - Implement Tensor Tile + Attention Engine + KV Controller in
    SystemVerilog.
  - Bring up a small decoder LLM end-to-end on FPGA.
  - Co-validate FPGA vs. simulator vs. C++ kernels.
- **Stage C — ASIC tape-out (year 2.5+):**
  - Edge profile first (smaller die, lower risk).
  - Server profile second.

## 9.6 Software stack on top of the ISA

- **LonghornAI compiler** — lowers a model graph (Phase 0 runtime) onto the
  Longhorn ISA, with kernel selection driven by Phase 8 dossiers.
- **LonghornAI runtime** — request scheduler, KV cache manager, continuous
  batching.
- **Bit-exact reference** — the same C++ kernels in this repository remain
  the golden oracle for both simulator and silicon.

## 9.7 Phase 9 — Acceptance criteria

- [ ] A complete `docs/silicon/architecture.md` describing the Longhorn
      microarchitecture in enough detail for an RTL team to start.
- [ ] A cycle-accurate simulator that runs the LonghornAI kernel suite
      bit-exactly.
- [ ] A roofline validation showing the simulator matches the Phase 8
      predictions within 10%.
- [ ] FPGA bring-up plan with milestones.

---

# Master Milestone Checklist

| Phase | Milestone | Target |
|---|---|---|
| 0 | Infrastructure complete; CI green on 3 OS × 3 compilers | M1 |
| 1 | Foundational kernels + GEMM ≥ 30% peak | M2 |
| 2 | Full attention stack incl. FlashDecoding | M3 |
| 3 | Paged KV + continuous batching + prefix cache | M4 |
| 4 | INT4/FP8/MXFP forward path on Llama-3-70B | M5 |
| 5 | Mixtral + DeepSeek-MoE forward | M6 |
| 6 | Speculative + Medusa + Eagle decoding | M7 |
| 7 | Mamba-2 + RWKV + hybrid Jamba forward | M8 |
| 8 | Per-kernel silicon dossiers + roofline | M9 |
| 9 | Architecture document + cycle-accurate sim | M10 |

---

# Performance Goals

These are the headline numbers. Each phase's bench harness reports against
them automatically.

## CPU reference (developer machines)

- **GEMM (FP32, large):** ≥ 30% of theoretical peak per core.
- **RMSNorm / Softmax (long row):** ≥ 70% of single-core memory bandwidth.
- **FlashAttention prefill (S=4096, FP16):** ≥ 50% of GEMM-equivalent
  peak.
- **FlashDecoding (S_kv=32k, B=1, FP16):** ≥ 60% of single-core memory
  bandwidth.

## Predicted Longhorn silicon (informational — set in Phase 8/9)

- **Tokens/sec/watt (Llama-3-70B INT4, server profile):** target 5×–10× over
  contemporary GPUs at equivalent context.
- **TTFT (Llama-3-70B INT4, S_in=2048):** target sub-200 ms at server profile.
- **ITL (Llama-3-70B INT4, decode):** target sub-20 ms steady-state.
- **Edge profile (Llama-3-8B Q4):** target ≥ 30 tok/s at < 5 W.

These numbers are revised every phase as more data lands.

---

# Benchmarking Strategy

- One harness, one report format, one shape-suite system.
- `bench/shapes/<model>.yaml` defines the shapes per model family.
- Every kernel reports `(shape, dtype, P50, P90, P99, FLOPS, GB/s, AI)` per
  run, written to `bench/results/<date>/<kernel>.csv`.
- A daily CI job runs the full suite on a reference machine and posts a diff
  vs. the previous day. Regression > 5% blocks merge.
- Per-phase headline plots:
  - Phase 1: tiled-vs-naive GEMM speedup; roofline.
  - Phase 2: SDPA vs. FlashAttention vs. FlashDecoding latency at varying S.
  - Phase 3: tokens/sec under continuous batching at varying request rate.
  - Phase 4: throughput vs. perplexity per quantization scheme.
  - Phase 5: MoE expert-load-balancing curves.
  - Phase 6: speculative-decoding acceptance vs. throughput.
  - Phase 7: Mamba/RWKV throughput at varying sequence length.

---

# Validation Strategy

- **Differential testing.** Every kernel is tested against its naive
  reference. No kernel is tested only against another optimized kernel.
- **Numerical tolerances** are documented per dtype in `docs/numerics.md`
  and enforced via `expect_close`.
- **End-to-end golden traces.** For each supported model, a fixed input →
  fixed output trace is checked into `tests/integration/golden/`. Forward
  passes must reproduce the trace bit-for-bit (FP32 paths) or within
  per-dtype tolerance (FP16/BF16/FP8/INT4 paths).
- **Stress tests.** Long-context (1M tokens), large-batch (1k+ requests),
  memory-pressure (KV swap), denormals, NaN/Inf handling.
- **Cross-architecture parity.** For each alternative architecture (Phase
  7), a small reference checkpoint is fully forward-validated against its
  upstream implementation.
- **Determinism.** Seeded RNG, no wall-clock dependence, no thread-count
  dependence in numerics (reductions are tree-reduced with a fixed
  reduction tree).

---

# Risks and Open Questions

- **Memory bandwidth dominance.** Decode performance is bound by KV-cache
  bandwidth. If the KV controller and on-the-fly dequant are not strong
  enough, no compute fabric will save the design. **Mitigation:** Phase 3
  is the deepest in the plan and is gated by a bandwidth-utilization
  acceptance criterion.
- **MoE all-to-all.** Multi-tile expert parallelism requires a strong NoC.
  **Mitigation:** the NoC is sized in Phase 8 from MoE bandwidth profiles
  before Phase 9 commits.
- **Long-context viability.** 1M-context attention is currently impractical
  at full precision. **Mitigation:** sparse-KV (H2O-style), MLA-style
  compression, and FlashDecoding are all in scope; the architecture must
  carry at least one to ship.
- **Architectural drift.** New post-Transformer architectures may render the
  attention engine over-specialized. **Mitigation:** Mamba-2 SSD already
  shares ≥ 80% of the attention datapath; the scan engine is the hedge.
- **Quantization fidelity.** INT4 / FP8 may break newer instruction-tuned
  models. **Mitigation:** keep FP8 (E4M3) as the default deployment dtype;
  INT4 is opt-in per model with per-model perplexity gates.

---

# Glossary

- **AI** — Arithmetic Intensity (FLOPs per byte of memory traffic).
- **GQA / MQA / MHA** — Grouped / Multi / Multi-head Attention.
- **KV cache** — stored Keys and Values from previous decode steps.
- **MoE** — Mixture of Experts.
- **MLA** — Multi-head Latent Attention (DeepSeek).
- **MX** — Microscaling (block-shared exponent low-precision).
- **RoPE** — Rotary Position Embedding.
- **SSD** — Structured State-space Duality (Mamba-2 reformulation).
- **SDPA** — Scaled Dot-Product Attention.
- **TTFT / ITL** — Time-To-First-Token / Inter-Token Latency.

---

*End of PLAN.md — the LonghornAI master plan: from a portable C++ kernel
library to the candidate Longhorn inference accelerator microarchitecture,
staged in nine deliberate phases.*
