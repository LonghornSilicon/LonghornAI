"""Silicon bring-up smoke harness (PLAN.md §8 M7 exit gate).

PLAN.md §8 M7 exit gate: **Silicon smoke + per-kernel correctness pass;
cross-target equivalence holds**. This module ships the smoke runner —
a curated set of per-kernel correctness checks that runs every registered
LonghornAI kernel under the lhsil backend and compares against the CPU
reference. The same harness powers the silicon bring-up playbook (PLAN.md
§5.1) when first silicon arrives: smoke → per-kernel correctness → end-to-
end model accuracy → performance characterization.

The CLI entry is :func:`longhornai.cli.silicon_smoke` (``lhai silicon-smoke``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..runtime import available_backends, dispatch, use_backend
from .equivalence import (
    EquivalenceReport,
    _to_array,
    assert_cross_target_equivalent,
)
from .tolerance import max_rel_error, tolerance_for


# --- smoke fixtures --------------------------------------------------------

def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def _gemm_fixture(rng):
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((32, 8)).astype(np.float32)
    return ("gemm", (a, b), {})


def _layernorm_fixture(rng):
    return ("layernorm", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _rmsnorm_fixture(rng):
    return ("rmsnorm", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _softmax_fixture(rng):
    return ("softmax", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _gelu_fixture(rng):
    return ("gelu", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _silu_fixture(rng):
    return ("silu", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _rope_fixture(rng):
    return ("rope", (rng.standard_normal((1, 4, 8)).astype(np.float32),), {})


def _reduce_fixture(rng):
    return ("reduce", (rng.standard_normal((4, 16)).astype(np.float32),), {})


def _embedding_fixture(rng):
    return (
        "embedding_lookup",
        (rng.standard_normal((32, 8)).astype(np.float32),
         np.array([[0, 5, 17, 31]], dtype=np.int64)),
        {},
    )


def _sdpa_fixture(rng):
    q = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    return ("sdpa", (q, k, v), {"causal": True})


def _flash_v2_fixture(rng):
    q = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    return ("flash_attention_v2", (q, k, v), {"causal": True, "block_q": 8, "block_kv": 8})


def _paged_attention_fixture(rng):
    B, S_q, H_q, H_kv, D = 1, 4, 4, 2, 8
    block_size = 4
    q = rng.standard_normal((B, S_q, H_q * D)).astype(np.float32)
    ck = rng.standard_normal((4, H_kv, block_size, D)).astype(np.float32)
    cv = rng.standard_normal((4, H_kv, block_size, D)).astype(np.float32)
    block_table = np.array([[0, 1, -1]], dtype=np.int32)
    seq_lens = np.array([S_q], dtype=np.int32)
    return (
        "paged_attention", (q, ck, cv),
        {"block_table": block_table, "seq_lens": seq_lens, "block_size": block_size,
         "num_q_heads": H_q, "num_kv_heads": H_kv, "head_dim": D, "causal": True},
    )


def _all_reduce_fixture(rng):
    a = rng.standard_normal((4, 4)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    return ("all_reduce", ([a, b],), {"op": "sum"})


def _moe_router_fixture(rng):
    x = rng.standard_normal((6, 16)).astype(np.float32)
    W = rng.standard_normal((16, 4)).astype(np.float32)
    return ("moe_router", (x, W), {})


def _gemm_w4a16_fixture(rng):
    from ..quantization import pack_int4, quantize_groupwise
    K, N = 64, 8
    W = rng.standard_normal((K, N)).astype(np.float32)
    q, params = quantize_groupwise(W, bits=4, group_size=32, axis=0)
    packed = pack_int4(q.astype(np.int8), axis=0)
    a = rng.standard_normal((4, K)).astype(np.float32)
    return (
        "gemm_w4a16", (a, packed),
        {"scale_b": params.scale, "group_size": 32, "K": K, "out_dtype": np.float32},
    )


def _sliding_window_fixture(rng):
    q = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 16, 8)).astype(np.float32)
    return ("sliding_window_attention", (q, k, v), {"window": 8})


def _gated_mlp_fixture(rng):
    H, I = 16, 32
    x = rng.standard_normal((4, H)).astype(np.float32)
    return (
        "gated_mlp", (x,),
        {"gate_weight": rng.standard_normal((H, I)).astype(np.float32),
         "up_weight": rng.standard_normal((H, I)).astype(np.float32),
         "down_weight": rng.standard_normal((I, H)).astype(np.float32),
         "activation": "silu"},
    )


SMOKE_FIXTURES: List[Callable[[Any], Tuple[str, tuple, dict]]] = [
    _gemm_fixture,
    _layernorm_fixture,
    _rmsnorm_fixture,
    _softmax_fixture,
    _gelu_fixture,
    _silu_fixture,
    _rope_fixture,
    _reduce_fixture,
    _embedding_fixture,
    _sdpa_fixture,
    _flash_v2_fixture,
    _paged_attention_fixture,
    _all_reduce_fixture,
    _moe_router_fixture,
    _gemm_w4a16_fixture,
    _sliding_window_fixture,
    _gated_mlp_fixture,
]


# --- runner ----------------------------------------------------------------

@dataclass
class SmokeOutcome:
    op: str
    backend: str
    passed: bool
    max_rel_error: float
    error: str | None = None


@dataclass
class SmokeReport:
    backend: str
    outcomes: List[SmokeOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    @property
    def num_failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.passed)

    def report(self) -> str:
        lines = [
            f"Silicon bring-up smoke — backend={self.backend}",
            f"  {len(self.outcomes)} ops checked, {self.num_failed} failure(s)",
        ]
        for o in self.outcomes:
            tag = "PASS" if o.passed else "FAIL"
            line = f"  [{tag}] {o.op:<28} max_rel_err={o.max_rel_error:.2e}"
            if o.error:
                line += f"  ({o.error})"
            lines.append(line)
        return "\n".join(lines)


def run_silicon_smoke(
    *, backend: str = "lhsil", reference: str = "cpu", seed: int = 0,
) -> SmokeReport:
    """Run every smoke fixture under ``backend`` and compare to ``reference``.

    The standard PLAN.md §5.1 silicon-bring-up sequence: power-on smoke →
    per-kernel correctness → end-to-end model accuracy → performance.
    This entry covers the first two for every registered op family.
    """
    rng = _rng(seed)
    outcomes: List[SmokeOutcome] = []
    for make_fixture in SMOKE_FIXTURES:
        op, args, kwargs = make_fixture(rng)
        try:
            tol = tolerance_for("float32")
            with use_backend(reference):
                ref_out = dispatch(op, *args, **kwargs)
            with use_backend(backend):
                got = dispatch(op, *args, **kwargs)
            ref_flat = _to_array(ref_out)
            got_flat = _to_array(got)
            ok = bool(np.allclose(
                ref_flat, got_flat,
                rtol=tol.rtol, atol=tol.atol, equal_nan=True,
            ))
            err = max_rel_error(got_flat, ref_flat)
            outcomes.append(SmokeOutcome(
                op=op, backend=backend, passed=ok, max_rel_error=err,
                error=None if ok else f"exceeds rtol={tol.rtol}",
            ))
        except Exception as exc:  # pragma: no cover - failure path
            outcomes.append(SmokeOutcome(
                op=op, backend=backend, passed=False,
                max_rel_error=float("inf"), error=repr(exc),
            ))
    return SmokeReport(backend=backend, outcomes=outcomes)


def run_full_cross_target_sweep(*, seed: int = 0) -> Dict[str, EquivalenceReport]:
    """Sweep every smoke fixture across every registered backend.

    Returns ``{op_name: EquivalenceReport}``. Each report covers all 5 pre-
    silicon backends (cpu / sim / rtl / fpga / lhsil); the keystone test for
    PLAN.md §5.2.
    """
    rng = _rng(seed)
    backends = available_backends()
    reports: Dict[str, EquivalenceReport] = {}
    for make_fixture in SMOKE_FIXTURES:
        op, args, kwargs = make_fixture(rng)
        reports[op] = assert_cross_target_equivalent(
            op, *args, dtype=np.float32, backends=backends, **kwargs,
        )
    return reports


__all__ = [
    "SmokeOutcome",
    "SmokeReport",
    "SMOKE_FIXTURES",
    "run_silicon_smoke",
    "run_full_cross_target_sweep",
]
