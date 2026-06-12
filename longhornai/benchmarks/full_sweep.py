"""Full performance characterization sweep (PLAN.md §8 M8).

The M8 exit gate requires "Production inference readiness criteria met
(§9)". Section 9 ranges from kernel-level (GEMM % of cuBLAS, FA parity,
roofline efficiency) to system-level (tokens/sec, latency SLO, scaling
efficiency). This module ships the *measurement* harness that drives
every relevant op family across a curated shape suite and emits a single
report; CI feeds it to the regression-hardening harness in
:mod:`longhornai.validation.regression`.

What's covered:

* GEMM (already in :mod:`benchmarks.shape_suite`) — wired through to the
  full report.
* Attention (SDPA, FlashAttention v2) — prefill + decode-shape sweeps.
* Paged attention (decode-shape: ``S_q == 1``).
* W4A16 GEMM (the production decode quant path).
* Decode tokens/sec (the system-level KPI from
  :mod:`benchmarks.decode_bench`).

The output is a list of :class:`PerfRow` records that
:mod:`longhornai.validation.regression` compares against the recorded
baseline.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from ..kernels import (
    flash_attention_v2,
    gemm,
    gemm_w4a16,
    paged_attention,
    paged_kv_alloc,
    paged_kv_append,
    sdpa,
)
from ..models import LlamaConfig, init_random_weights
from ..quantization import pack_int4, quantize_groupwise
from .decode_bench import run_decode_benchmark
from .shape_suite import GEMM_SHAPE_SUITE, run_gemm_suite


@dataclass(frozen=True)
class PerfRow:
    """One performance measurement."""

    family: str               # "gemm" | "attention" | "paged_attention" | ...
    op: str                   # specific op name
    shape: str                # short label
    dtype: str
    latency_ms: float
    achieved_tflops: float | None = None
    achieved_gbps: float | None = None
    tokens_per_second: float | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def report_line(self) -> str:
        bits = [f"{self.family:<14} {self.op:<24} {self.shape:<24} "
                f"{self.dtype:<8} {self.latency_ms:9.3f} ms"]
        if self.achieved_tflops is not None:
            bits.append(f" {self.achieved_tflops:7.3f} TFLOP/s")
        if self.tokens_per_second is not None:
            bits.append(f" {self.tokens_per_second:9.1f} tok/s")
        return "".join(bits)


def _bench(fn: Callable[[], Any], *, warmup: int = 1, trials: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --- per-family sweeps ----------------------------------------------------

def _sweep_gemm(*, dtype) -> List[PerfRow]:
    # Pick the subset of the canonical GEMM shape suite that runs quickly
    # on the CPU reference; lhsil's full sweep covers everything.
    wanted = {"square_1k", "decode_proj_4k", "decode_mlp_down_7b",
              "qwen_attn_o_proj", "llama7b_qkv_proj"}
    cases = [c for c in GEMM_SHAPE_SUITE if c.name in wanted]
    results = run_gemm_suite(dtype=dtype, cases=cases, warmup=1, trials=2)
    return [
        PerfRow(
            family="gemm", op="gemm",
            shape=f"{r.name}({r.M}x{r.K}x{r.N})", dtype=r.dtype,
            latency_ms=r.latency_ms,
            achieved_tflops=r.achieved_tflops,
            achieved_gbps=r.achieved_gbps,
            extras={"is_compute_bound": r.is_compute_bound,
                    "roofline_efficiency": r.roofline_efficiency},
        )
        for r in results
    ]


def _sweep_attention(*, dtype) -> List[PerfRow]:
    rng = np.random.default_rng(0)
    rows: List[PerfRow] = []
    # Prefill (compute-bound) and decode (memory-bound) shapes.
    cases = [
        ("prefill_short", 1, 4, 128, 16),
        ("prefill_long",  1, 8, 256, 16),
        ("decode",        4, 8,   1, 16),
    ]
    for name, B, H, S, D in cases:
        q = rng.standard_normal((B, H, S, D)).astype(dtype)
        k = rng.standard_normal((B, H, S, D)).astype(dtype)
        v = rng.standard_normal((B, H, S, D)).astype(dtype)
        for op_name, fn in (
            ("sdpa", lambda q=q, k=k, v=v: sdpa(q, k, v, causal=True)),
            ("flash_attention_v2",
             lambda q=q, k=k, v=v: flash_attention_v2(q, k, v, causal=True)),
        ):
            t = _bench(fn)
            # FLOPs ≈ 4·B·H·S²·D for prefill (dominant: QK^T + scores·V).
            flops = 4 * B * H * S * S * D
            rows.append(PerfRow(
                family="attention", op=op_name, shape=name, dtype=str(np.dtype(dtype)),
                latency_ms=t * 1e3,
                achieved_tflops=(flops / t) / 1e12 if t > 0 else 0.0,
                extras={"B": B, "H": H, "S": S, "D": D},
            ))
    return rows


def _sweep_paged_attention(*, dtype) -> List[PerfRow]:
    rng = np.random.default_rng(0)
    rows: List[PerfRow] = []
    # Decode-shape: S_q=1 across a populated paged KV.
    block_size = 16
    H_q, H_kv, D = 4, 2, 16
    for label, seq_len in (("decode_S128", 128), ("decode_S512", 512)):
        n_blocks = (seq_len + block_size - 1) // block_size
        ck, cv = paged_kv_alloc(num_blocks=n_blocks, block_size=block_size,
                                 num_kv_heads=H_kv, head_dim=D, dtype=dtype)
        block_table = np.arange(n_blocks, dtype=np.int32).reshape(1, -1)
        # Fill the cache.
        full_k = rng.standard_normal((1, seq_len, H_kv, D)).astype(dtype)
        full_v = rng.standard_normal((1, seq_len, H_kv, D)).astype(dtype)
        paged_kv_append(ck, cv, full_k, full_v,
                         block_table=block_table,
                         seq_lens=np.zeros(1, dtype=np.int32),
                         block_size=block_size)
        q = rng.standard_normal((1, 1, H_q * D)).astype(dtype)
        seq_lens_after = np.array([seq_len], dtype=np.int32)

        def call(ck=ck, cv=cv, q=q, sla=seq_lens_after,
                 bt=block_table, bs=block_size):
            return paged_attention(
                q, ck, cv, block_table=bt, seq_lens=sla, block_size=bs,
                num_q_heads=H_q, num_kv_heads=H_kv, head_dim=D, causal=True,
            )

        t = _bench(call)
        rows.append(PerfRow(
            family="paged_attention", op="paged_attention",
            shape=label, dtype=str(np.dtype(dtype)),
            latency_ms=t * 1e3,
            extras={"S_kv": seq_len, "H_q": H_q, "H_kv": H_kv, "D": D},
        ))
    return rows


def _sweep_quant(*, dtype) -> List[PerfRow]:
    rng = np.random.default_rng(0)
    rows: List[PerfRow] = []
    for label, M, K, N in (("decode_4k", 1, 1024, 1024),
                            ("prefill_4k", 256, 1024, 1024)):
        W = rng.standard_normal((K, N)).astype(np.float32)
        q, params = quantize_groupwise(W, bits=4, group_size=64, axis=0)
        packed = pack_int4(q.astype(np.int8), axis=0)
        a = rng.standard_normal((M, K)).astype(dtype)

        def call(a=a, packed=packed, scale=params.scale, K=K):
            return gemm_w4a16(a, packed, scale_b=scale, group_size=64,
                                K=K, out_dtype=dtype)

        t = _bench(call)
        flops = 2 * M * N * K
        rows.append(PerfRow(
            family="quant", op="gemm_w4a16", shape=label,
            dtype=str(np.dtype(dtype)),
            latency_ms=t * 1e3,
            achieved_tflops=(flops / t) / 1e12 if t > 0 else 0.0,
            extras={"M": M, "K": K, "N": N, "group_size": 64},
        ))
    return rows


def _sweep_decode_tokens_per_sec(*, dtype) -> List[PerfRow]:
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
    )
    weights = init_random_weights(config, dtype=dtype, seed=0)
    rows: List[PerfRow] = []
    for batch in (1, 4):
        result = run_decode_benchmark(
            config=config, weights=weights,
            num_requests=batch, prompt_len=8, max_new_tokens=12,
            label=f"decode_b{batch}",
        )
        rows.append(PerfRow(
            family="serving", op="decode", shape=f"batch={batch}",
            dtype=str(np.dtype(dtype)),
            latency_ms=result.wall_time_s * 1e3,
            tokens_per_second=result.tokens_per_second,
            extras={"iterations": result.iterations,
                    "decode_tokens": result.decode_tokens},
        ))
    return rows


# --- top-level entry ------------------------------------------------------

def run_full_sweep(*, dtype=np.float32) -> List[PerfRow]:
    """Run every per-family sweep and return all rows."""
    rows: List[PerfRow] = []
    rows.extend(_sweep_gemm(dtype=dtype))
    rows.extend(_sweep_attention(dtype=dtype))
    rows.extend(_sweep_paged_attention(dtype=dtype))
    rows.extend(_sweep_quant(dtype=dtype))
    rows.extend(_sweep_decode_tokens_per_sec(dtype=dtype))
    return rows


def format_full_sweep(rows: List[PerfRow]) -> str:
    header = (f"{'family':<14} {'op':<24} {'shape':<24} {'dtype':<8} "
              f"{'lat (ms)':>10}     extras")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(r.report_line())
    return "\n".join(lines)


# --- baseline I/O ---------------------------------------------------------

_BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "full_sweep_baseline.json"


def baseline_path() -> pathlib.Path:
    return _BASELINE_PATH


def save_full_sweep_baseline(rows: List[PerfRow], *, note: str = "") -> None:
    payload = {
        "schema": 1,
        "note": note or (
            "Full performance characterization baseline (M8). CPU NumPy "
            "reference; production targets land on silicon (PLAN.md §9)."
        ),
        "rows": [asdict(r) for r in rows],
    }
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2))


def load_full_sweep_baseline() -> Dict[str, Any]:
    if not _BASELINE_PATH.exists():
        return {"schema": 1, "rows": []}
    return json.loads(_BASELINE_PATH.read_text())


def compare_full_sweep(
    rows: List[PerfRow], baseline: Dict[str, Any], *,
    regression_factor: float = 1.5,
) -> List[Dict[str, Any]]:
    """Compare a fresh sweep to the recorded baseline."""
    by_key = {(r["family"], r["op"], r["shape"], r["dtype"]): r
              for r in baseline.get("rows", [])}
    out = []
    for r in rows:
        key = (r.family, r.op, r.shape, r.dtype)
        prev = by_key.get(key)
        prev_lat = prev["latency_ms"] if prev else None
        regressed = bool(prev_lat is not None and
                          r.latency_ms > prev_lat * regression_factor)
        out.append({
            "family": r.family, "op": r.op, "shape": r.shape, "dtype": r.dtype,
            "latency_ms": r.latency_ms, "baseline_ms": prev_lat,
            "regressed": regressed,
            "ratio": (r.latency_ms / prev_lat) if prev_lat else None,
        })
    return out


__all__ = [
    "PerfRow",
    "run_full_sweep",
    "format_full_sweep",
    "save_full_sweep_baseline",
    "load_full_sweep_baseline",
    "compare_full_sweep",
    "baseline_path",
]
