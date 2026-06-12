"""Decode tokens/sec benchmark (PLAN.md §8 M4).

PLAN.md §8 M4 exit gate: a published tokens/sec baseline for end-to-end
Llama decode under continuous batching. This module drives the
:class:`ContinuousBatchingScheduler` over a toy Llama for a configurable
number of requests and reports:

* aggregate tokens/sec (decode tokens divided by wall time),
* per-request mean tokens/sec,
* prefill vs decode time split,
* iterations and pool occupancy.

The recorded baseline lives in ``benchmarks/decode_baseline.json`` —
populated by ``lhai decode-bench --save-baseline``.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from ..models import LlamaConfig, LlamaWeights, init_random_weights
from ..runtime import (
    ContinuousBatchingScheduler,
    Request,
    SchedulerConfig,
    SchedulerStats,
)


@dataclass(frozen=True)
class DecodeBenchmarkResult:
    """Aggregated decode-throughput metrics."""

    label: str
    backend: str
    num_requests: int
    prompt_len: int
    max_new_tokens: int
    iterations: int
    prefill_tokens: int
    decode_tokens: int
    wall_time_s: float
    tokens_per_second: float

    def report(self) -> str:
        return (
            f"{self.label}\n"
            f"  backend         : {self.backend}\n"
            f"  requests        : {self.num_requests}  "
            f"(prompt_len={self.prompt_len}, max_new={self.max_new_tokens})\n"
            f"  iterations      : {self.iterations}\n"
            f"  prefill tokens  : {self.prefill_tokens}\n"
            f"  decode tokens   : {self.decode_tokens}\n"
            f"  wall time       : {self.wall_time_s * 1e3:9.2f} ms\n"
            f"  tokens / sec    : {self.tokens_per_second:9.1f}"
        )


def run_decode_benchmark(
    *,
    config: LlamaConfig,
    weights: LlamaWeights,
    num_requests: int = 4,
    prompt_len: int = 8,
    max_new_tokens: int = 16,
    sched_config: Optional[SchedulerConfig] = None,
    backend: str = "cpu",
    label: str = "llama-toy-decode",
    seed: int = 0,
) -> DecodeBenchmarkResult:
    """Drive the continuous-batching scheduler and time the decode loop.

    The same prompt length and ``max_new_tokens`` is used for every request
    (uniform-shape simplifies measurement). The scheduler is allowed to
    interleave prefill + decode normally; total decode tokens divided by
    wall time gives the baseline tokens/sec.
    """
    from ..runtime import use_backend

    sc = sched_config or SchedulerConfig(
        max_batch_size=max(num_requests, 1),
        num_blocks=max(64, 4 * num_requests),
        block_size=8,
        max_blocks_per_request=max(8, 2 * (prompt_len + max_new_tokens) // 8),
        cache_dtype=weights.embed_tokens.dtype,
    )

    scheduler = ContinuousBatchingScheduler(weights, config, sc)
    rng = np.random.default_rng(seed)
    for i in range(num_requests):
        prompt = rng.integers(0, config.vocab_size, size=(prompt_len,)).astype(np.int64)
        scheduler.add_request(
            Request(
                request_id=f"r{i}",
                prompt_ids=prompt,
                max_new_tokens=max_new_tokens,
            )
        )

    with use_backend(backend):
        t0 = time.perf_counter()
        stats: SchedulerStats = scheduler.run_until_done()
        dt = time.perf_counter() - t0

    total_decode = stats.decode_tokens
    tps = total_decode / dt if dt > 0 else float("inf")
    return DecodeBenchmarkResult(
        label=label,
        backend=backend,
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        iterations=stats.iterations,
        prefill_tokens=stats.prefill_tokens,
        decode_tokens=total_decode,
        wall_time_s=dt,
        tokens_per_second=tps,
    )


# --- Baseline I/O -----------------------------------------------------------

_BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "decode_baseline.json"


def baseline_path() -> pathlib.Path:
    return _BASELINE_PATH


def load_baseline() -> Dict[str, dict]:
    if not _BASELINE_PATH.exists():
        return {"schema": 1, "results": []}
    return json.loads(_BASELINE_PATH.read_text())


def save_baseline(results: List[DecodeBenchmarkResult], *, note: str = "") -> None:
    payload = {
        "schema": 1,
        "note": note or (
            "CPU NumPy reference baseline; production tokens/sec lands on "
            "silicon (PLAN.md §9.2)."
        ),
        "results": [asdict(r) for r in results],
    }
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2))


__all__ = [
    "DecodeBenchmarkResult",
    "run_decode_benchmark",
    "load_baseline",
    "save_baseline",
    "baseline_path",
]
