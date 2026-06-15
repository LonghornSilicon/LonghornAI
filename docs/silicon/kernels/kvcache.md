# Paged KV cache (`src/kernels/kvcache.{hpp,cpp}`, `paged_attention.{hpp,cpp}`, `kvcache_quant.{hpp,cpp}`)

## What it computes
- `paged_kv_append` — write new K/V into block-table-indexed positions
  inside a `[num_blocks, n_kv, block_size, head_dim]` pool.
- `paged_attention` — single-request attention that walks a per-request
  block table to fetch K/V; equivalent to SDPA over a contiguous gather
  of the indexed blocks.
- `paged_attention_batched` — same, n_requests at a time.
- `paged_kv_append_q8`, `paged_attention_q8` — INT8 KV variant: per
  `(block, head, slot)` fp32 scale; on-the-fly dequant on the read path.

## Reference path
`paged_attention` validated bit-exact against `sdpa`; `paged_attention_q8`
within MSE < 1e-3 vs the fp32 path on unit-variance random data.

## Memory traffic
- Append: `2 · n_kv · seq_new · head_dim` writes per layer.
- Attention (decode): `2 · Lk · n_kv · head_dim` reads per layer per
  request — the dominant DRAM traffic of inference at scale.
- INT8 variant: KV reads halve (1 byte/elem); plus `2 · Lk · n_kv`
  bytes of scale reads (negligible).

## Arithmetic intensity
Attention AI is unchanged by paging. The relevant silicon question is
**effective bandwidth utilisation**: an indirect block walk must hit
near-peak HBM throughput.

## Roofline class
**Memory-bound** for decode. The KV controller's prefetcher is the
single most-leveraged piece of decode silicon.

## Engine assignment
**KV Cache Controller.** Dedicated block:
1. Block-table walker (logical → physical).
2. Multi-stream prefetcher.
3. On-the-fly dequant (INT4 / INT8 / FP8 → FP16 / BF16).
4. Append path with quantization on write.
5. Eviction / migration descriptors driven by DMA.

Coherent with the Attention Engine: single-cycle handoff of (K, V, mask)
tiles.

## SRAM working set
- Block: `block_size · n_kv · head_dim · bytes_per_elem` per block.
  For block_size=16, n_kv=8, head_dim=128, FP16: 32 KiB per block.
  Two blocks resident at any time gives a streaming-friendly layout
  (one being attended, one being prefetched).
- Block table: `4 · max_blocks_per_request` bytes. Trivial.

## DMA pattern
**Indirected block-grain streaming.** Block table provides physical
block ids; controller prefetches the next block before the current
finishes.

## Datapath dtype requirements
- Storage: FP16 / BF16 / FP8 / INT8 / INT4 (per scheme).
- Read-path dequant: produces FP16 / BF16 (whatever attention consumes).
- Append: input dtype → storage dtype with quantization.

## Open silicon questions
- Block size sweet spot: 16 minimises fragmentation, 64 maximises
  prefetch efficiency. Need workload data.
- Whether to support virtual-paged migration to host memory (vLLM's
  swap path) — only useful for very long contexts.
