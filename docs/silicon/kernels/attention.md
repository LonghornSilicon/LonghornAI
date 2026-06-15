# Attention (`src/kernels/attention.{hpp,cpp}`)

## What it computes
- `sdpa`: `softmax(QKᵀ / √d + bias) V`.
- `flash_attention`: same, online-softmax tiled, no materialised score
  matrix. Q-outer / K-inner partitioning is FlashAttention-2 shape.
- `flash_decode_attention`: split-K + online-softmax merge. The decode
  hot path when Lk ≫ Lq.
- `cross_attention`: `sdpa` with `causal=false`.
- Mask compositions: `causal`, `window` (sliding), `chunk`, additive
  `bias` (covers ALiBi + tree masks + arbitrary structured masks).

## Reference path
`sdpa` is the golden numerics anchor; `flash_attention` and
`flash_decode_attention` must match it within tolerance.

## Memory traffic
For one `(batch, head, query)`:
- Reads: `Lk · d` (K) + `Lk · d` (V).
- Writes: `d` (output).
- Plus per-step Q `(d)` already in registers.

KV bandwidth dominates decode: every step rereads the full cache.

## Arithmetic intensity
- Score: `2 · Lq · Lk · d` FLOPs against `(Lq + Lk) · d` bytes →
  AI ~ d/2 (= 32 for d=64).
- Output: same AI shape.
- Combined attention AI ≈ d/2 (~16–32 for typical heads).

## Roofline class
- Prefill (`Lq = Lk` long): **balanced** (the score matmul tile fits
  on the Tensor Unit).
- Decode (`Lq = 1, Lk` large): **memory-bound**, dominated by the KV
  read. This is the entire reason the KV controller exists.

## Engine assignment
**Attention Engine.** Fused pipeline:
1. Q tile loader.
2. K tile streamer (from KV controller).
3. QKᵀ multiply with scale, mask, bias add.
4. Online softmax (`m`, `ℓ`) state register.
5. V tile streamer + weighted sum.
6. (m, ℓ, O) carry register.

## SRAM working set
Per attention block: a Q tile (`Mq · d` FP16 ~ 4 KiB), a K tile
(`block_k · d` FP16 ~ 4 KiB), a V tile (same), the (m, ℓ, O) state
(`Mq · (1 + 1 + d)` FP32 ~ 16 KiB). **~30 KiB per active (Lq tile,
head) pair.**

## DMA pattern
- Q: **streamed** from upstream projection.
- K, V: **streamed via KV controller** (block-table indirect; on
  decode the access pattern is one row per cache position per step).
- O: **streamed** to downstream projection.

## Datapath dtype requirements
- Q, K, V: BF16 / FP16 (or FP8 in aggressive mode).
- Score lanes: **FP32** (online-softmax max-rescale needs the
  headroom).
- (m, ℓ, O) accumulators: **FP32**.

## Open silicon questions
- Tile size for FlashDecoding's `splits`: trade off between merge
  overhead and per-split parallelism.
- Whether MQA / GQA motivates a separate cheaper engine (the K/V
  bandwidth is already amortised).
