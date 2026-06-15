# LonghornAI Silicon Analysis (Phase 8)

This directory is the bridge between the C++ kernel library and the
Longhorn microarchitecture (Phase 9). It contains:

- **`roofline.md`** — the roofline framing and per-kernel bound classes.
- **`kernel_engine_map.md`** — the consolidated `kernel → engine → SRAM →
  DMA → dtype` table that drives accelerator block sizing.
- **`kernels/<name>.md`** — per-kernel dossiers, one per family. Each
  dossier follows the 9-section template defined in `PLAN.md` §3.

These documents are generated from the same kernel inventory the test
suite exercises; numbers come from `./build/longhorn_bench --roofline`.

## How to read a dossier

Each `kernels/*.md` answers, in order:

1. **What it computes** — math.
2. **Reference path** — file + function in the repo.
3. **Memory traffic** — bytes read / written / reused per call.
4. **Arithmetic intensity** — FLOPs/byte; pinned to a measured number
   from the bench suite when applicable.
5. **Roofline class** — compute / memory / balanced (against a ridge of
   ~10 FLOPs/byte for a representative x86 core; the threshold scales
   linearly to whatever target's peak FLOPS / peak BW ratio).
6. **Engine assignment** — which Longhorn block executes this in
   silicon.
7. **SRAM working set** — size of the on-chip scratchpad needed to keep
   the kernel from spilling to HBM mid-tile.
8. **DMA pattern** — strided / scatter-gather / transposed-tile / etc.
9. **Open silicon questions** — known unknowns that block Phase 9
   decisions.
