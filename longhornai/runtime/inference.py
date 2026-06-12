"""Production runtime packaging — :class:`LonghornInference`.

PLAN.md §8 M8 deliverable: "production runtime packaging". This module is
the public, stable, documented API surface a serving stack imports — the
analog of vLLM's ``LLM`` class. It composes :class:`LlamaConfig` +
:class:`LlamaWeights` + :class:`ContinuousBatchingScheduler` under one
constructor and exposes ``generate`` / ``stream`` entry points.

Internals are deliberately thin:
* ``LonghornInference(weights, config, sched_config=...)`` — wires up the
  scheduler + paged KV pool.
* ``.generate(prompts, max_new_tokens=..., backend="cpu", ...)`` — accepts a
  list of token-ID lists, runs continuous batching to completion, returns
  the generated tokens per prompt.
* ``.stream(prompts, ...)`` — iterator that yields token-by-token; enables
  streaming serving without changing the underlying scheduler.
* ``.shutdown()`` — releases the paged pool. (Tests use ``with`` blocks.)

PLAN.md §10.3 (Future Vision: Longhorn Serving Runtime) is the long-term
home of this surface; M8 lands the in-process API so downstream products
have a stable target while the standalone server is being built.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from ..models.llama import LlamaConfig, LlamaWeights
from ..runtime import use_backend
from ..runtime.scheduler import (
    ContinuousBatchingScheduler,
    Request,
    SchedulerConfig,
    SchedulerStats,
)


@dataclass
class GenerationResult:
    """One prompt's generation outcome."""

    request_id: str
    output_ids: List[int]
    prompt_length: int
    generated_length: int


class LonghornInference:
    """Public production API for serving Llama-class models.

    Wraps the :class:`ContinuousBatchingScheduler` so callers don't have to
    manage paged-KV state directly. Supports any backend the runtime knows
    about (cpu / sim / rtl / fpga / lhsil) via the ``backend=`` argument
    on :meth:`generate` / :meth:`stream`.
    """

    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        *,
        sched_config: Optional[SchedulerConfig] = None,
        sampler: Optional[Callable[[np.ndarray], int]] = None,
    ) -> None:
        self.weights = weights
        self.config = config
        self._sched_config = sched_config or SchedulerConfig(
            max_batch_size=8,
            num_blocks=256,
            block_size=16,
            max_blocks_per_request=64,
            cache_dtype=weights.embed_tokens.dtype,
            enable_prefix_cache=True,
        )
        self._sampler = sampler
        self._scheduler: Optional[ContinuousBatchingScheduler] = None

    def _ensure_scheduler(self) -> ContinuousBatchingScheduler:
        if self._scheduler is None:
            self._scheduler = ContinuousBatchingScheduler(
                self.weights, self.config, self._sched_config,
                sampler=self._sampler,
            )
        return self._scheduler

    # --- batched generation ----------------------------------------------

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        backend: str = "cpu",
        request_id_prefix: str = "lh",
    ) -> List[GenerationResult]:
        """Run greedy continuous-batching generation on ``prompts``.

        Each entry of ``prompts`` is a sequence of token IDs. Returns one
        :class:`GenerationResult` per prompt in the input order.
        """
        sched = self._ensure_scheduler()
        # Reset any state from a previous call.
        sched.waiting.clear()
        sched.finished.clear()
        sched.stats = SchedulerStats()

        ordered_ids: List[str] = []
        for i, prompt in enumerate(prompts):
            rid = f"{request_id_prefix}-{i}"
            ordered_ids.append(rid)
            sched.add_request(Request(
                request_id=rid,
                prompt_ids=np.asarray(prompt, dtype=np.int64),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            ))

        with use_backend(backend):
            sched.run_until_done()

        finished_by_id = {r.request_id: r for r in sched.finished}
        out: List[GenerationResult] = []
        for rid, prompt in zip(ordered_ids, prompts):
            req = finished_by_id[rid]
            out.append(GenerationResult(
                request_id=rid,
                output_ids=list(req.output_ids),
                prompt_length=len(prompt),
                generated_length=len(req.output_ids),
            ))
        return out

    # --- streaming generation --------------------------------------------

    def stream(
        self,
        prompt: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        backend: str = "cpu",
        request_id: str = "lh-stream",
    ) -> Iterator[int]:
        """Yield generated tokens one at a time for a single prompt.

        The scheduler runs one ``step`` per yielded token; serving stacks
        can wrap this in an SSE / WebSocket emitter without changing the
        underlying scheduling loop.
        """
        sched = self._ensure_scheduler()
        sched.waiting.clear()
        sched.finished.clear()
        sched.stats = SchedulerStats()
        sched.add_request(Request(
            request_id=request_id,
            prompt_ids=np.asarray(prompt, dtype=np.int64),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ))
        emitted = 0
        with use_backend(backend):
            while sched.has_pending_work:
                sched.step()
                # Each step appends at most one new token per active request
                # (modulo the prefill iteration which produces one too).
                # Find our request — it's either still in slots or in finished.
                for r in sched.finished:
                    while r.request_id == request_id and emitted < len(r.output_ids):
                        yield int(r.output_ids[emitted])
                        emitted += 1
                    if r.request_id == request_id:
                        return
                for slot_req in sched.slots:
                    if slot_req is not None and slot_req.request_id == request_id:
                        while emitted < len(slot_req.output_ids):
                            yield int(slot_req.output_ids[emitted])
                            emitted += 1
                        break

    # --- lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        """Release the scheduler's paged KV pool."""
        self._scheduler = None

    def __enter__(self) -> "LonghornInference":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


@contextmanager
def inference_session(
    weights: LlamaWeights, config: LlamaConfig, **kwargs,
) -> Iterator[LonghornInference]:
    """Convenience context-manager wrapping :class:`LonghornInference`."""
    inf = LonghornInference(weights, config, **kwargs)
    try:
        yield inf
    finally:
        inf.shutdown()


__all__ = [
    "GenerationResult",
    "LonghornInference",
    "inference_session",
]
