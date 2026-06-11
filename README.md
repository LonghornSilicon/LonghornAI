# LonghornAI

Canonical AI inference kernel library and runtime substrate for Longhorn Silicon
accelerators. See [`PLAN.md`](PLAN.md) for the full engineering roadmap.

## Status — M1 (Foundation)

The first milestone is implemented: the repository scaffold, the backend-dispatch
runtime, the **Phase-1 foundational compute kernels** on a portable CPU (NumPy)
reference backend, the numerical validation harness, the benchmarking/roofline
harness, an auto-tuning driver, and quantization primitives.

| Area | Delivered in M1 |
|------|-----------------|
| Kernels (Phase 1) | GEMM · batched GEMM · grouped GEMM · tensor contraction · LayerNorm · RMSNorm · Softmax · GELU (erf/tanh) · SiLU · RoPE · embedding lookup · reductions |
| Runtime | Backend registry + dispatch; `use_backend` context switching (CPU today; sim/RTL/FPGA/silicon per PLAN.md §2.2) |
| Validation | Per-dtype tolerance policy, float64 golden references, differential testing harness |
| Benchmarking | Latency, throughput, FLOPS/bandwidth utilization, arithmetic intensity, roofline efficiency |
| Compiler | Grid-search auto-tuning driver (seed of the compiler stack) |
| Quantization | Symmetric INT quantize/dequantize primitives (foundation for Phase 4) |
| Models | Llama-style SwiGLU MLP block composed from kernels |

## Install

```bash
pip install -e .          # runtime (numpy)
pip install -e '.[dev]'   # + pytest / ruff / mypy
```

## Quickstart

```python
import numpy as np
import longhornai as lh

a = np.random.randn(64, 128).astype("float32")
b = np.random.randn(128, 32).astype("float32")
c = lh.gemm(a, b)         # dispatches to the active backend
```

## CLI

```bash
lhai backends                       # list execution backends
lhai kernels                        # list kernels with goldens
lhai validate --dtype float16       # differential-test all kernels vs goldens
lhai bench gemm -m 4096 -k 4096 -n 4096 --dtype float16
```

## Tests

```bash
pytest -q
```

## Layout

```
longhornai/
├── kernels/        # Phase-1 kernel front-ends (dispatch wrappers)
├── runtime/        # backend abstraction + operator dispatch
├── backends/       # per-target impls (cpu; sim/rtl/fpga/lhsil to come)
├── validation/     # tolerance policy, golden refs, differential testing
├── benchmarks/     # cost models + roofline harness
├── compiler/       # auto-tuning driver (compiler-stack seed)
├── quantization/   # quantize/dequantize primitives
├── models/         # end-to-end blocks composed from kernels
└── cli/            # `lhai` command-line interface
```
