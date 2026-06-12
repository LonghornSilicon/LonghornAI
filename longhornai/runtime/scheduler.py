"""Continuous-batching scheduler (PLAN.md §3 Phase 3 / §8 M4).

The scheduler is the heart of the M4 decode/serving stack. It owns:

* a fixed-size paged KV pool (allocated once at startup),
* a list of physical block IDs available for assignment,
* a queue of waiting requests and a list of active (in-flight) requests,
* (M5) an optional content-addressable :class:`PrefixCache` for shared-prefix
  KV-block reuse — multiple requests with the same prompt prefix share
  physical blocks instead of recomputing them.

and advances the system one **iteration** at a time:

1. Admit as many waiting requests as fit the dynamic-batch budget. With
   the prefix cache enabled, walk each admitted request's prompt
   block-by-block and reuse cached physical blocks where the prefix
   matches.
2. Run a uniform-shape **prefill** call for the just-admitted requests,
   sample one token each. Suffix-only when the prefix cache covers part
   of the prompt.
3. Run a uniform-shape **decode** step for every active request, sample
   one token each.
4. Append the new tokens to each request's output buffer and check for
   completion (max-new-tokens, EOS).

PLAN.md §8 M4 exit gate: "E2E Llama decode under continuous batching on
FPGA" + a published tokens/sec baseline. PLAN.md §8 M5 adds **prefix
caching** and **speculative decoding** to the same loop.

The default sampler is **greedy argmax** so the scheduler is deterministic
and the equivalence harness can compare backends bit-by-bit. Top-k /
nucleus sampling lands with the serving runtime productization (PLAN.md
§10.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..models.llama import (
    LlamaConfig,
    LlamaWeights,
    PagedKVState,
    alloc_paged_kv_state,
    llama_decode_step,
    llama_prefill,
)


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Request:
    """One in-flight (or waiting) generation request."""

    request_id: str
    prompt_ids: np.ndarray
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    state: RequestState = RequestState.WAITING
    output_ids: List[int] = field(default_factory=list)
    slot: Optional[int] = None

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED


@dataclass
class SchedulerStats:
    """Aggregated metrics across one or more scheduler iterations."""

    iterations: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    requests_completed: int = 0
    prefix_blocks_reused: int = 0   # M5 prefix-cache hit count


@dataclass
class SchedulerConfig:
    """Pool sizing + dynamic-batch knobs."""

    max_batch_size: int = 8
    max_blocks_per_request: int = 64
    num_blocks: int = 256
    block_size: int = 16
    cache_dtype: np.dtype = np.float16
    enable_prefix_cache: bool = False  # M5 — opt in for shared-prefix reuse


# --- prefix cache (M5) ---------------------------------------------------

class PrefixCache:
    """Content-addressable KV-block cache for shared-prefix reuse.

    Keys chain along the prompt: ``key_i = (key_{i-1}, tuple_of_block_tokens)``.
    Each cache entry holds ``[physical_block_id, refcount]``. A block is
    freed back to the scheduler's pool only when its refcount reaches zero,
    so concurrent requests sharing a prefix can read the same physical
    block safely.

    The hash chain ensures *contiguous prefix* reuse only — a block at
    logical position N reuses the cached block iff every prior block also
    matched. Discontiguous reuse would invalidate the chain.
    """

    _ROOT_KEY: Tuple = ()

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._cache: Dict[Tuple, List] = {}

    @classmethod
    def root_key(cls) -> Tuple:
        return cls._ROOT_KEY

    @staticmethod
    def chain(parent: Tuple, tokens) -> Tuple:
        """Compute the cache key for ``tokens`` chained off ``parent``."""
        token_tuple = tuple(int(t) for t in tokens)
        return (parent, token_tuple)

    def acquire(self, key: Tuple) -> Optional[int]:
        """Increment refcount on a cached block; return its physical id, or None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        entry[1] += 1
        return entry[0]

    def insert(self, key: Tuple, block_id: int) -> None:
        """Insert a fresh block (refcount = 1)."""
        self._cache[key] = [block_id, 1]

    def release(self, key: Tuple) -> Optional[int]:
        """Decrement refcount; return ``block_id`` if it became reclaimable."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        entry[1] -= 1
        if entry[1] <= 0:
            del self._cache[key]
            return entry[0]
        return None

    @property
    def size(self) -> int:
        return len(self._cache)


class ContinuousBatchingScheduler:
    """Iteration-level scheduler over a paged KV pool.

    Construction allocates the pool once. ``add_request`` queues a request;
    ``step`` runs one iteration (admit + prefill + decode). The scheduler
    keeps slots assigned for the lifetime of a request — eviction is done
    by setting ``state = FINISHED`` and reclaiming the slot for the next
    waiting request.
    """

    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        sched_config: Optional[SchedulerConfig] = None,
        *,
        sampler: Optional[Callable[[np.ndarray], int]] = None,
    ) -> None:
        self.weights = weights
        self.config = config
        self.sched_config = sched_config or SchedulerConfig()
        self.sampler = sampler or self._greedy_argmax

        sc = self.sched_config
        self.state = alloc_paged_kv_state(
            config,
            batch_size=sc.max_batch_size,
            num_blocks=sc.num_blocks,
            block_size=sc.block_size,
            max_blocks_per_seq=sc.max_blocks_per_request,
            dtype=sc.cache_dtype,
        )
        self.free_blocks: List[int] = list(range(sc.num_blocks))
        self.slots: List[Optional[Request]] = [None] * sc.max_batch_size
        self.slot_blocks: List[List[int]] = [[] for _ in range(sc.max_batch_size)]
        self.waiting: List[Request] = []
        self.finished: List[Request] = []
        self.stats = SchedulerStats()

        # Prefix cache state (M5).
        self.prefix_cache: Optional[PrefixCache] = (
            PrefixCache(sc.block_size) if sc.enable_prefix_cache else None
        )
        # Per-slot list of cache keys held (each refcounted on acquire) and
        # cached prefix length (a multiple of block_size).
        self.slot_cache_keys: List[List[Tuple]] = [
            [] for _ in range(sc.max_batch_size)
        ]
        self.slot_cached_len: List[int] = [0] * sc.max_batch_size
        # Per-slot suffix block keys *not yet inserted* into the cache —
        # they go in once the block is fully written.
        self.slot_pending_block_keys: List[List[Tuple]] = [
            [] for _ in range(sc.max_batch_size)
        ]

    # --- request lifecycle -------------------------------------------------

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting.append(request)

    @property
    def has_pending_work(self) -> bool:
        return bool(self.waiting) or any(s is not None for s in self.slots)

    @property
    def active_slots(self) -> List[int]:
        return [i for i, s in enumerate(self.slots) if s is not None]

    # --- one iteration -----------------------------------------------------

    def step(self) -> SchedulerStats:
        """Advance one iteration: admit + prefill + decode."""
        admitted = self._admit_pending()
        if admitted:
            self._run_prefill(admitted)
        decoding = [s for s in self.active_slots
                    if self.slots[s] is not None
                    and self.slots[s].state == RequestState.DECODING]
        if decoding:
            self._run_decode(decoding)
        self.stats.iterations += 1
        return self.stats

    def run_until_done(self, *, max_iterations: int = 1024) -> SchedulerStats:
        for _ in range(max_iterations):
            if not self.has_pending_work:
                break
            self.step()
        return self.stats

    # --- admission + dynamic batching --------------------------------------

    def _budget_for_admission(self) -> int:
        free_slots = sum(1 for s in self.slots if s is None)
        if free_slots == 0 or not self.waiting:
            return 0
        return min(free_slots, len(self.waiting))

    def _walk_prefix_cache(
        self, prompt: np.ndarray,
    ) -> Tuple[List[Tuple], List[int]]:
        """Walk ``prompt`` block-by-block, acquiring cached physical blocks.

        Returns ``(acquired_keys, acquired_block_ids)``. Stops at the first
        miss (contiguous-prefix policy) and never includes the *last* block
        of the prompt — at least one block must remain to be prefilled so
        we have a place to sample logits from.
        """
        if self.prefix_cache is None:
            return [], []
        bs = self.sched_config.block_size
        # Maximum number of blocks we *could* reuse — leave at least one to
        # prefill (the kernel needs to produce logits for the suffix).
        max_reusable = max(0, (len(prompt) // bs) - 1) if len(prompt) % bs == 0 \
            else len(prompt) // bs
        keys: List[Tuple] = []
        blocks: List[int] = []
        parent = PrefixCache.root_key()
        for i in range(max_reusable):
            tokens = prompt[i * bs : (i + 1) * bs]
            key = PrefixCache.chain(parent, tokens)
            block_id = self.prefix_cache.acquire(key)
            if block_id is None:
                break
            keys.append(key)
            blocks.append(block_id)
            parent = key
        return keys, blocks

    def _pending_block_keys(
        self, prompt: np.ndarray, cached_count: int,
    ) -> List[Tuple]:
        """Compute the cache keys for the *complete* suffix blocks we are
        about to fill. Partial trailing blocks aren't cached until completed
        in a later decode step."""
        if self.prefix_cache is None:
            return []
        bs = self.sched_config.block_size
        n_complete = len(prompt) // bs
        if n_complete <= cached_count:
            return []
        # Re-derive the parent chain from the cached prefix.
        parent = PrefixCache.root_key()
        for i in range(cached_count):
            parent = PrefixCache.chain(parent, prompt[i * bs : (i + 1) * bs])
        keys: List[Tuple] = []
        for i in range(cached_count, n_complete):
            key = PrefixCache.chain(parent, prompt[i * bs : (i + 1) * bs])
            keys.append(key)
            parent = key
        return keys

    def _admit_pending(self) -> List[int]:
        """Move admissable waiting requests into free slots; return slot indices."""
        budget = self._budget_for_admission()
        admitted: List[int] = []
        for _ in range(budget):
            req = self.waiting[0]
            worst_case = self._blocks_needed(
                len(req.prompt_ids) + req.max_new_tokens
            )
            if worst_case > self.sched_config.max_blocks_per_request:
                raise RuntimeError(
                    f"request {req.request_id!r} needs up to {worst_case} "
                    f"blocks; SchedulerConfig.max_blocks_per_request="
                    f"{self.sched_config.max_blocks_per_request}"
                )
            if worst_case > self.sched_config.num_blocks:
                raise RuntimeError(
                    f"request {req.request_id!r} needs up to {worst_case} "
                    f"blocks; pool exhausted "
                    f"(num_blocks={self.sched_config.num_blocks})"
                )

            # Prefix-cache walk: acquire any cached prefix blocks first so
            # we only allocate fresh storage for the suffix.
            cached_keys, cached_blocks = self._walk_prefix_cache(req.prompt_ids)
            cached_len = len(cached_blocks) * self.sched_config.block_size
            total_needed = self._blocks_needed(len(req.prompt_ids))
            fresh_needed = total_needed - len(cached_blocks)
            if len(self.free_blocks) < fresh_needed:
                # Release any prefix blocks we acquired so we don't pin them.
                for key in cached_keys:
                    self.prefix_cache.release(key)
                break  # transient — wait for blocks to free

            slot = self._first_free_slot()
            if slot is None:
                for key in cached_keys:
                    self.prefix_cache.release(key)
                break

            fresh_blocks = [self.free_blocks.pop(0) for _ in range(fresh_needed)]
            all_blocks = cached_blocks + fresh_blocks
            self.slot_blocks[slot] = list(fresh_blocks)  # only ours to free
            self.state.block_table[slot, : len(all_blocks)] = all_blocks
            self.state.block_table[slot, len(all_blocks):] = -1
            self.state.seq_lens[slot] = cached_len
            self.slot_cache_keys[slot] = list(cached_keys)
            self.slot_cached_len[slot] = cached_len
            self.slot_pending_block_keys[slot] = self._pending_block_keys(
                req.prompt_ids, len(cached_keys),
            )
            req.slot = slot
            req.state = RequestState.PREFILLING
            self.slots[slot] = req
            self.waiting.pop(0)
            admitted.append(slot)
            if cached_blocks:
                self.stats.prefix_blocks_reused += len(cached_blocks)
        return admitted

    def _first_free_slot(self) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s is None:
                return i
        return None

    def _blocks_needed(self, num_tokens: int) -> int:
        bs = self.sched_config.block_size
        return (num_tokens + bs - 1) // bs

    # --- prefill / decode --------------------------------------------------

    def _run_prefill(self, admitted_slots: List[int]) -> None:
        """Prefill the just-admitted requests. Group by (suffix_len, offset)."""
        grouped: Dict[Tuple[int, int], List[int]] = {}
        for s in admitted_slots:
            req = self.slots[s]
            suffix_len = len(req.prompt_ids) - self.slot_cached_len[s]
            key = (suffix_len, self.slot_cached_len[s])
            grouped.setdefault(key, []).append(s)

        for (suffix_len, offset), slots in grouped.items():
            self._prefill_group(suffix_len, offset, slots)

    def _prefill_group(self, suffix_len: int, position_offset: int,
                       slots: List[int]) -> None:
        original = self.state
        scratch = PagedKVState(
            block_size=original.block_size,
            cache_k=original.cache_k,
            cache_v=original.cache_v,
            block_table=original.block_table[slots].copy(),
            seq_lens=np.full((len(slots),), position_offset, dtype=np.int32),
        )
        prompt_suffixes = np.stack(
            [self.slots[s].prompt_ids[position_offset:] for s in slots], axis=0,
        )
        logits = llama_prefill(
            prompt_suffixes, self.weights, self.config,
            state=scratch, position_offset=position_offset,
        )

        for k, s in enumerate(slots):
            original.seq_lens[s] = scratch.seq_lens[k]

        # Insert any newly-completed full blocks into the prefix cache. A
        # block is "full" when seq_lens has crossed past its end position.
        if self.prefix_cache is not None:
            self._insert_completed_blocks(slots)

        for k, s in enumerate(slots):
            tok = self.sampler(logits[k, -1])
            req = self.slots[s]
            req.output_ids.append(int(tok))
            req.state = RequestState.DECODING
            self.stats.prefill_tokens += suffix_len
            self._maybe_finish(s, tok)

    def _insert_completed_blocks(self, slots: List[int]) -> None:
        """For each just-prefilled slot, insert any newly-complete blocks."""
        bs = self.sched_config.block_size
        for s in slots:
            seq_len = int(self.state.seq_lens[s])
            n_complete = seq_len // bs
            already_inserted = self.slot_cached_len[s] // bs
            new_complete = max(0, n_complete - already_inserted)
            if new_complete <= 0 or not self.slot_pending_block_keys[s]:
                continue
            for _ in range(min(new_complete, len(self.slot_pending_block_keys[s]))):
                key = self.slot_pending_block_keys[s].pop(0)
                block_idx = already_inserted
                our_phys = int(self.state.block_table[s, block_idx])
                # Race-safe insert: if another request already cached this
                # exact prefix block, free our duplicate and redirect the
                # block table to the canonical cached copy. The cache holds
                # at most one physical block per key.
                existing = self.prefix_cache.acquire(key)
                if existing is None:
                    # Fresh insert — cache takes our just-written block.
                    self.prefix_cache.insert(key, our_phys)
                    if our_phys in self.slot_blocks[s]:
                        self.slot_blocks[s].remove(our_phys)
                else:
                    if existing != our_phys:
                        # Duplicate; return our copy to the free pool.
                        if our_phys in self.slot_blocks[s]:
                            self.slot_blocks[s].remove(our_phys)
                        self.free_blocks.append(our_phys)
                        self.state.block_table[s, block_idx] = existing
                self.slot_cache_keys[s].append(key)
                already_inserted += 1
                self.slot_cached_len[s] += bs

    def _run_decode(self, slots: List[int]) -> None:
        token_ids = np.array(
            [self.slots[s].output_ids[-1] for s in slots], dtype=np.int64,
        )
        scratch = PagedKVState(
            block_size=self.state.block_size,
            cache_k=self.state.cache_k,
            cache_v=self.state.cache_v,
            block_table=self.state.block_table[slots].copy(),
            seq_lens=self.state.seq_lens[slots].copy(),
        )
        for k, s in enumerate(slots):
            self._ensure_capacity(s, scratch, k)
        logits = llama_decode_step(
            token_ids, self.weights, self.config, state=scratch,
        )
        for k, s in enumerate(slots):
            self.state.seq_lens[s] = scratch.seq_lens[k]
            self.state.block_table[s] = scratch.block_table[k]

        for k, s in enumerate(slots):
            tok = self.sampler(logits[k])
            self.slots[s].output_ids.append(int(tok))
            self.stats.decode_tokens += 1
            self._maybe_finish(s, tok)

    def _ensure_capacity(self, slot: int, scratch: PagedKVState, k: int) -> None:
        bs = self.sched_config.block_size
        next_pos = int(scratch.seq_lens[k])
        block_idx = next_pos // bs
        if scratch.block_table[k, block_idx] >= 0:
            return
        if not self.free_blocks:
            raise RuntimeError(
                f"paged KV pool exhausted while extending slot {slot}; "
                f"raise SchedulerConfig.num_blocks"
            )
        new_block = self.free_blocks.pop(0)
        self.slot_blocks[slot].append(new_block)
        scratch.block_table[k, block_idx] = new_block

    def _maybe_finish(self, slot: int, last_token: int) -> None:
        req = self.slots[slot]
        eos_hit = req.eos_token_id is not None and last_token == req.eos_token_id
        max_hit = len(req.output_ids) >= req.max_new_tokens
        if eos_hit or max_hit:
            req.state = RequestState.FINISHED
            self.finished.append(req)
            # Release any prefix-cache references this slot held; freed-by-
            # cache blocks come back to our free pool.
            if self.prefix_cache is not None:
                for key in self.slot_cache_keys[slot]:
                    freed = self.prefix_cache.release(key)
                    if freed is not None:
                        self.free_blocks.append(freed)
            # Reclaim the slot's privately-owned blocks.
            for blk in self.slot_blocks[slot]:
                self.free_blocks.append(blk)
            self.slot_blocks[slot] = []
            self.slot_cache_keys[slot] = []
            self.slot_cached_len[slot] = 0
            self.slot_pending_block_keys[slot] = []
            self.state.block_table[slot] = -1
            self.state.seq_lens[slot] = 0
            self.slots[slot] = None
            self.stats.requests_completed += 1

    # --- samplers ----------------------------------------------------------

    @staticmethod
    def _greedy_argmax(logits: np.ndarray) -> int:
        return int(np.argmax(logits))


__all__ = [
    "Request",
    "RequestState",
    "SchedulerConfig",
    "SchedulerStats",
    "ContinuousBatchingScheduler",
    "PrefixCache",
]
