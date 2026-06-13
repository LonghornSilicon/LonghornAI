# LonghornAI — Inference Kernel Library Plan

This document is the staged roadmap and design rationale for **LonghornAI**, a
portable, pure **C++17** library of the common AI-inference kernels that appear
across modern LLM architectures (Llama, Mistral, Qwen, Gemma, Phi, and friends).
It is written for Longhorn Silicon as the canonical, readable reference
implementation of every operator an inference stack needs.

Each phase is a self-contained unit of work with explicit deliverables, files,
the math each kernel computes, and acceptance criteria, so future work can be
picked up without re-reading anything else.

The **design philosophy** is non-negotiable across every phase: mathematical
transparency, readable kernels, one operator + one naive reference + one test
per kernel, and portability over cleverness. Every kernel is plain C++ that
compiles and runs identically on Windows, Linux, and macOS.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Design Principles](#2-design-principles)
3. [Repository Layout](#3-repository-layout)
4. [Build & Portability](#4-build--portability)
5. [Kernel Roadmap](#5-kernel-roadmap)
   - [Phase 0 — Scaffold & Build](#phase-0--scaffold--build)
   - [Phase 1 — Foundational Compute](#phase-1--foundational-compute)
   - [Phase 2 — Attention](#phase-2--attention)
6. [Testing & Validation](#6-testing--validation)
7. [Benchmarking](#7-benchmarking)
8. [Out of Scope](#8-out-of-scope)

---

## 1. Vision

LonghornAI is the single, readable, portable C++ reference for the inference
kernels that every modern LLM forward pass is built from. It is not a GPU
library, not a framework, and not a serving stack — it is the **operator set**:
GEMM, normalization, activations, positional embeddings, softmax, reductions,
and the attention family, each implemented once, clearly, in standard C++.

The library is **inference-first** and **forward-only**. It defines the
numerical semantics of each operator (via a naive reference) and provides a
faster, cache-aware production implementation that is proven equivalent to that
reference. Downstream Longhorn projects — compilers, serving runtimes, silicon
validation suites — can read these kernels to understand exactly what each
operator computes, and can use them as the golden behavioural contract.

---

## 2. Design Principles

- **Pure C++17, no exceptions to portability.** Kernels use only the C++
  standard library. No CUDA, no GPU code, no SIMD intrinsics, no third-party
  compute dependencies. The library builds with MSVC, GCC, and Clang on
  Windows, Linux, and macOS with zero platform-specific code paths.
- **One kernel, one reference, one test.** Every operator ships with (1) a
  dead-simple naive reference that defines correct numerics, (2) a faster
  cache-aware implementation, and (3) a test that proves they agree within a
  documented tolerance. No kernel is "done" without all three.
- **Readability over cleverness.** A reader should be able to see the math.
  Blocking, accumulation order, and numerical-stability tricks are commented
  where they matter and nowhere else.
- **Measure, then optimize.** Optimizations (loop blocking, optional
  multithreading) are added only after the naive path is correct, and only with
  a benchmark that shows the win.
- **FP32 is the source of truth.** FP16/BF16 are supported via a portable
  software half/bfloat16 helper type; their tolerances are looser and
  documented.

---

## 3. Repository Layout

```
LonghornAI/
  CMakeLists.txt            cross-platform build, C++17 only, no CUDA
  PLAN.md                   this roadmap
  README.md
  src/
    kernels/                one <name>.hpp + <name>.cpp per kernel family
      dtypes.hpp            portable half / bfloat16 helper types
      tensor_view.hpp       lightweight non-owning shape/stride view
      gemm.hpp/.cpp         blocked GEMM (+ batched / grouped)
      normalization.hpp/.cpp LayerNorm, RMSNorm
      activation.hpp/.cpp   GELU (erf/tanh), SiLU, SwiGLU, GeGLU
      softmax.hpp/.cpp      stable row softmax
      reduction.hpp/.cpp    sum / max / mean primitives
      rope.hpp/.cpp         rotary position embeddings
      embedding.hpp/.cpp    embedding gather
      attention.hpp/.cpp    SDPA, flash-style, MHA/MQA/GQA, KV-cache
  tests/                    GoogleTest suite, kernel-vs-naive-reference
```

Each kernel header declares a host-callable function with a stable signature
operating on raw pointers plus shape metadata (and/or a small `TensorView`).
The `.cpp` carries both the naive reference (used by tests as the golden) and
the optimized implementation.

---

## 4. Build & Portability

- **CMake >= 3.20**, `project(LonghornAI LANGUAGES CXX)` — C++ only, no CUDA
  language enabled.
- **C++17**, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`.
- **No platform-specific APIs.** Only the C++ standard library.
- **GoogleTest via `FetchContent`** so the test suite builds on any OS without
  a system install.
- **Per-compiler warning flags**, guarded so each toolchain gets the right
  ones:
  - MSVC: `/W3`
  - GCC / Clang: `-Wall -Wextra`
- **Optional multithreading** behind `-DLONGHORN_ENABLE_OPENMP=ON`. When off
  (the default), every kernel runs correctly single-threaded. OpenMP is never
  required to build or pass tests.

Build is the standard CMake flow on every platform:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## 5. Kernel Roadmap

Kernels are sequenced so foundational compute lands first, then attention builds
on softmax/GEMM/RoPE. Every kernel ships with a naive reference, an optimized
implementation, and a differential test before it is considered done.

### Phase 0 — Scaffold & Build

Goal: a building, testable skeleton with the shared helpers every kernel needs.

**Deliverables**

- `CMakeLists.txt` per §4 (CXX-only, GoogleTest via FetchContent, OpenMP option).
- `src/kernels/dtypes.hpp`: portable `half` and `bfloat16` types with
  `to_float` / `from_float` conversions (pure integer/bit math, no hardware
  dependency).
- `src/kernels/tensor_view.hpp`: lightweight non-owning view (data pointer +
  shape + strides) and row-major index helpers.
- `tests/` harness: a tolerance policy (per-dtype absolute/relative bounds), a
  random-tensor generator with a fixed seed, and an `expect_close` comparator.

**Acceptance**

- `cmake --build` succeeds on Windows/Linux/macOS with MSVC/GCC/Clang.
- A trivial placeholder test runs green via CTest.
- Half/bfloat16 round-trip conversions are within their documented tolerance.

Status: planned.

### Phase 1 — Foundational Compute

Goal: the bedrock dense + elementwise + reduction kernels that an LLM block is
assembled from.

| Kernel | Computes | Notes |
|--------|----------|-------|
| **GEMM** | `C = alpha * A @ B + beta * C` | Blocked/tiled, cache-friendly, FP32 accumulate; naive triple-loop reference. |
| **Batched GEMM** | per-batch `C_i = A_i @ B_i` | Uniform-shape batching for projections. |
| **Grouped GEMM** | variable-shape groups | Ragged batches; foundation for future MoE. |
| **LayerNorm** | `(x - mean) / sqrt(var + eps) * g + b` | Welford-stable mean/variance, FP32 reduction. |
| **RMSNorm** | `x / sqrt(mean(x^2) + eps) * g` | Llama/Qwen/Mistral norm. |
| **Softmax** | `exp(x - max) / sum(exp(x - max))` | Max-subtracted for stability; per-row. |
| **GELU** | erf and tanh variants | Exact-match goldens to the standard formulas. |
| **SiLU / SwiGLU / GeGLU** | `x * sigmoid(x)`, gated MLP variants | Gated-MLP activation paths. |
| **RoPE** | rotary embedding | Interleaved & half-rotation layouts; NTK/linear scaling. |
| **Embedding lookup** | gather rows by id | Optional scale; large-vocab friendly. |
| **Reductions** | sum / max / mean | Axis reductions backing norms and softmax. |

**Files:** `gemm.hpp/.cpp`, `normalization.hpp/.cpp`, `activation.hpp/.cpp`,
`softmax.hpp/.cpp`, `rope.hpp/.cpp`, `embedding.hpp/.cpp`, `reduction.hpp/.cpp`,
each with matching tests in `tests/`.

**Acceptance (exit gate)**

- Every Phase-1 kernel matches its naive reference within the per-dtype
  tolerance on the random + edge-case suites (empty, single-element,
  non-aligned, large rows).
- Blocked GEMM beats the naive triple-loop on the shape suite in Release.
- Softmax and the norms are numerically stable on large-magnitude inputs.

Status: planned.

### Phase 2 — Attention

Goal: the attention family, built on Phase-1 softmax/GEMM/RoPE.

| Kernel | Computes | Notes |
|--------|----------|-------|
| **SDPA** | `softmax(Q Kᵀ / sqrt(d) + mask) V` | Reference fused attention; the correctness anchor for the rest. |
| **FlashAttention-style** | tiled, online-softmax attention | No materialized score matrix; running max/sum rescaling; numerical parity with SDPA. |
| **MHA / MQA / GQA** | head-config variants | Full multi-head, single-KV-head (MQA), and grouped-KV (GQA) routing. |
| **KV-cache** | append + gather | Append new K/V; gather over cached positions for decode. |

Causal and non-causal masking are supported; the flash-style kernel must match
the SDPA reference to within tolerance on both.

**Files:** `attention.hpp/.cpp` plus tests in `tests/`.

**Acceptance (exit gate)**

- Flash-style attention matches the SDPA reference within tolerance on causal
  and non-causal suites across MHA/MQA/GQA head configs.
- KV-cache append+gather produces identical results to full recompute over the
  same sequence.

Status: planned.

---

## 6. Testing & Validation

- **GoogleTest** suite under `tests/`, one test file per kernel family.
- **Naive reference as golden.** Each kernel is validated against its own
  simple, obviously-correct reference implementation — never against another
  optimized kernel.
- **Per-dtype tolerances.** FP32 tight (relative ~1e-5); FP16/BF16 looser and
  documented in the test harness; comparisons use combined absolute + relative
  bounds.
- **Edge cases.** Empty tensors, single-element rows, non-power-of-two and
  non-aligned shapes, long sequences/large vocab, and large-magnitude inputs
  for stability.
- **Determinism.** Fixed-seed random inputs so failures reproduce exactly.

---

## 7. Benchmarking

Lightweight and optional — correctness first, performance second.

- **Latency / throughput** per kernel over a shape suite drawn from real model
  configs (hidden sizes, head dims, sequence lengths).
- **Tiled-vs-naive GEMM speedup** reported as the headline Phase-1 perf number.
- **Attention parity** (flash-style vs. SDPA) reported for latency and numerics.

Benchmarks are reproducible (warm-up + steady-state, report distribution not
just mean) but are not a merge gate in this scope.

---

## 8. Out of Scope

To keep the core a focused, portable C++ kernel set, the following are
explicitly **not** part of this plan and are left as future work:

- CUDA / GPU / accelerator backends.
- Training, backward passes, and optimizers.
- Quantization (INT8/INT4, AWQ/GPTQ/SmoothQuant).
- Mixture-of-Experts (router, dispatch, expert parallelism).
- Paged attention and serving-level scheduling (continuous/dynamic batching).
- Multi-GPU / distributed inference and collectives.
- RTL / FPGA / silicon backends and cross-target equivalence harnesses.
- Python bindings and end-to-end model graphs.

*End of PLAN.md — a portable, pure C++ inference kernel library for Longhorn Silicon.*
