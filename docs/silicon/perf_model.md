# Longhorn Performance Model

The analytical performance model is the Stage A simulator from
[`roadmap.md`](roadmap.md). It takes a kernel description (engine
choice, FLOPs, bytes read/written, dtype) and a `LonghornConfig`
(per-engine peak FLOPS, on-chip and off-chip bandwidth) and returns
a predicted execution time plus a roofline class.

## Model

For each kernel:

```
t_compute = FLOPs / peak_lane_FLOPs[engine, dtype]
t_memory  = bytes / effective_bandwidth[memory_tier, kernel_pattern]

t_predicted = max(t_compute, t_memory)
bound = (t_compute > t_memory) ? compute : memory
```

The model is **roofline-accurate**, not cycle-accurate. It assumes:

- Linear scaling with lane count (true while bank conflicts and NoC
  contention are absent).
- Effective bandwidth = peak × utilisation factor (typically 0.7
  for HBM, 0.5 for LPDDR).
- No pipeline stall modelling — that comes at Stage B (FPGA, RTL).

## Peak rates

The model ships two configurations matching the architecture
document (`architecture.md` §12):

### Edge profile

| Engine | Peak | Notes |
|---|---|---|
| Tensor Unit | 32 × 32 × 2 GHz × 2 = 4 TFLOPS (FP16) | 2× for INT8, 4× for INT4 |
| Vector Unit | 64 lanes × 2 GHz × 2 = 256 GFLOPS | FP32 |
| Norm Engine | 16 lanes × 2 GHz = 32 G ops/s | FP32 reductions |
| Scan Engine | 64 channels × 2 GHz = 128 G ops/s | FP32 |
| L1 SRAM | 1 TB/s per tile | software-managed |
| L2 SRAM | 500 GB/s shared | |
| LPDDR5X | 100 GB/s × 0.5 = 50 GB/s effective | |

### Server profile

| Engine | Peak | Notes |
|---|---|---|
| Tensor Unit | 64 × 64 × 2 GHz × 2 = 16 TFLOPS (FP16) per tile × 8 tiles = 128 TFLOPS | |
| Vector Unit | 64 lanes × 2 GHz × 2 = 256 GFLOPS per tile | |
| Norm / Scan / etc. | scaled per tile count | |
| L1 SRAM | 1 TB/s per tile (8 TB/s aggregate) | |
| L2 SRAM | 500 GB/s shared | |
| HBM3e × 4 | 4 TB/s × 0.7 = 2.8 TB/s effective | |

## Validation

`tests/test_perf_model.cpp` runs the kernel inventory through the
model and asserts:

1. Every kernel's bound classification matches its Phase 8 dossier
   (`docs/silicon/kernels/<name>.md`).
2. Predicted execution time on the bench shape suite tracks the
   roofline-derived expectation within 10% (see `architecture.md` §13
   for the contract).
