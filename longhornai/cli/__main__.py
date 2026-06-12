"""``lhai`` CLI entry point.

Subcommands:

    lhai backends                 List registered execution backends.
    lhai kernels                  List Phase-1 kernels and their goldens.
    lhai validate [--dtype ...]   Differential-test every kernel vs its golden.
    lhai bench gemm [-m -k -n]    Benchmark a single GEMM and print roofline metrics.
    lhai bench-suite              Run the canonical GEMM shape suite.
    lhai decode-bench             Run the continuous-batching decode benchmark (M4).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .. import kernels, runtime
from ..benchmarks import benchmark, cost_for
from ..benchmarks.decode_bench import (
    load_baseline as load_decode_baseline,
    run_decode_benchmark,
    save_baseline as save_decode_baseline,
)
from ..benchmarks.shape_suite import (
    GEMM_SHAPE_SUITE,
    compare_to_baseline,
    format_suite_report,
    load_baseline,
    run_gemm_suite,
    save_baseline,
)
from ..models import LlamaConfig, init_random_weights
from ..validation import REFERENCES, differential_test


def _cmd_backends(_args) -> int:
    active = runtime.get_backend().name
    for name in runtime.available_backends():
        b = runtime.get_backend(name)
        marker = "*" if name == active else " "
        print(f" {marker} {name:<8} {len(b.ops()):>3} ops   {b.description}")
    print("\n(* = active backend)")
    return 0


def _cmd_kernels(_args) -> int:
    for op in sorted(REFERENCES):
        print(f"  {op}")
    print(f"\n{len(REFERENCES)} kernels with golden references.")
    return 0


# Differential cases: op -> (kernel callable, factory(dtype) -> args, kwargs).
def _validation_cases(dtype):
    rng = np.random.default_rng(0)

    def arr(*shape):
        return rng.standard_normal(shape).astype(dtype)

    return [
        ("gemm", kernels.gemm, (arr(64, 128), arr(128, 32)), {}),
        ("batched_gemm", kernels.batched_gemm, (arr(4, 16, 32), arr(4, 32, 8)), {}),
        ("layernorm", kernels.layernorm, (arr(32, 64),), {}),
        ("rmsnorm", kernels.rmsnorm, (arr(32, 64),), {}),
        ("softmax", kernels.softmax, (arr(32, 64),), {}),
        ("gelu", kernels.gelu, (arr(32, 64),), {}),
        ("silu", kernels.silu, (arr(32, 64),), {}),
        ("rope", kernels.rope, (arr(2, 8, 16),), {}),
        ("reduce", kernels.reduce, (arr(32, 64),), {}),
        ("sdpa", kernels.sdpa, (arr(1, 2, 16, 8), arr(1, 2, 16, 8), arr(1, 2, 16, 8)),
         {"causal": True}),
        ("flash_attention_v1", kernels.flash_attention_v1,
         (arr(1, 2, 16, 8), arr(1, 2, 16, 8), arr(1, 2, 16, 8)),
         {"causal": True, "block_q": 8, "block_kv": 8}),
    ]


def _cmd_validate(args) -> int:
    dtype = np.dtype(args.dtype)
    failures = 0
    for op, fn, fargs, fkwargs in _validation_cases(dtype):
        res = differential_test(op, fn, *fargs, dtype=dtype, **fkwargs)
        print(res)
        failures += int(not res.passed)
    print(f"\n{'OK' if failures == 0 else 'FAILED'}: "
          f"{failures} failure(s) on dtype={dtype.name}")
    return 1 if failures else 0


def _cmd_bench(args) -> int:
    if args.kernel != "gemm":
        print(f"bench: only 'gemm' supported in M1, got '{args.kernel}'", file=sys.stderr)
        return 2
    m, k, n = args.m, args.k, args.n
    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    cost = cost_for("gemm", dtype=dtype, m=m, k=k, n=n)
    result = benchmark(lambda: kernels.gemm(a, b), cost, name=f"gemm[{m}x{k}x{n}] {dtype.name}")
    print(result.report())
    return 0


def _cmd_bench_suite(args) -> int:
    dtype = np.dtype(args.dtype)
    cases = list(GEMM_SHAPE_SUITE)
    if args.quick:
        # Tiny subset for smoke-test runs; keep 3 representative cases.
        wanted = {"square_1k", "decode_proj_4k", "llama7b_qkv_proj"}
        cases = [c for c in cases if c.name in wanted]
    results = run_gemm_suite(
        dtype=dtype, cases=cases, warmup=args.warmup, trials=args.trials
    )
    print(format_suite_report(results))
    if args.save_baseline:
        save_baseline(results)
        print(f"\nbaseline saved with {len(results)} shape(s).")
        return 0
    base = load_baseline()
    if base.get("shapes"):
        cmp = compare_to_baseline(results, base, regression_factor=args.regression_factor)
        regs = [r for r in cmp if r["regressed"]]
        if regs:
            print(f"\n{len(regs)} shape(s) regressed > {args.regression_factor}× baseline:")
            for r in regs:
                print(f"  {r['name']}: {r['latency_ms']:.3f} ms vs {r['baseline_ms']:.3f} ms"
                      f"  ({r['ratio']:.2f}×)")
            return 1
    return 0


def _cmd_decode_bench(args) -> int:
    """Run continuous-batching decode and emit a tokens/sec report."""
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    weights = init_random_weights(config, dtype=np.dtype(args.dtype), seed=args.seed)
    result = run_decode_benchmark(
        config=config, weights=weights,
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        label=args.label,
        seed=args.seed,
    )
    print(result.report())
    if args.save_baseline:
        save_decode_baseline([result])
        print(f"\nbaseline saved.")
    return 0


def _cmd_silicon_smoke(args) -> int:
    """Run the M7 silicon-bring-up smoke harness."""
    from ..validation import run_full_cross_target_sweep, run_silicon_smoke

    report = run_silicon_smoke(
        backend=args.backend, reference=args.reference, seed=args.seed,
    )
    print(report.report())
    rc = 0 if report.passed else 1
    if args.cross_target:
        print("\nCross-target equivalence sweep across all registered backends:")
        sweep = run_full_cross_target_sweep(seed=args.seed)
        all_pass = True
        for op, eq in sweep.items():
            tag = "PASS" if eq.passed else "FAIL"
            print(f"  [{tag}] {op}")
            if not eq.passed:
                all_pass = False
        if not all_pass:
            rc = 1
    return rc


def _cmd_full_sweep(args) -> int:
    """Run the M8 full performance characterization sweep."""
    from ..benchmarks.full_sweep import (
        format_full_sweep,
        run_full_sweep,
        save_full_sweep_baseline,
    )
    rows = run_full_sweep(dtype=np.dtype(args.dtype))
    print(format_full_sweep(rows))
    if args.save_baseline:
        save_full_sweep_baseline(rows)
        print(f"\nbaseline saved with {len(rows)} row(s).")
    return 0


def _cmd_regression_check(args) -> int:
    """Run the M8 regression gate across all baselines."""
    from ..validation import run_regression_check
    report = run_regression_check(regression_factor=args.regression_factor)
    print(report.summary())
    return 0 if report.passed else 1


def _cmd_readiness(args) -> int:
    """Run the M8 production-readiness scorecard."""
    from ..validation.readiness import format_readiness_card, run_readiness_card

    card = run_readiness_card()
    print(format_readiness_card(card))
    return 0 if card.passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lhai", description="LonghornAI kernel toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backends", help="list execution backends").set_defaults(func=_cmd_backends)
    sub.add_parser("kernels", help="list kernels").set_defaults(func=_cmd_kernels)

    p_val = sub.add_parser("validate", help="differential-test kernels vs goldens")
    p_val.add_argument("--dtype", default="float32", help="working dtype (default float32)")
    p_val.set_defaults(func=_cmd_validate)

    p_bench = sub.add_parser("bench", help="benchmark a kernel")
    p_bench.add_argument("kernel", help="kernel to benchmark (gemm)")
    p_bench.add_argument("-m", type=int, default=4096)
    p_bench.add_argument("-k", type=int, default=4096)
    p_bench.add_argument("-n", type=int, default=4096)
    p_bench.add_argument("--dtype", default="float16")
    p_bench.set_defaults(func=_cmd_bench)

    p_suite = sub.add_parser("bench-suite", help="run the GEMM shape suite (PLAN.md §4.3)")
    p_suite.add_argument("--dtype", default="float16")
    p_suite.add_argument("--warmup", type=int, default=2)
    p_suite.add_argument("--trials", type=int, default=5)
    p_suite.add_argument("--quick", action="store_true",
                         help="run a 3-shape smoke subset")
    p_suite.add_argument("--save-baseline", action="store_true",
                         help="overwrite gemm_baseline.json with these results")
    p_suite.add_argument("--regression-factor", type=float, default=1.5,
                         help="latency regression threshold (multiplicative)")
    p_suite.set_defaults(func=_cmd_bench_suite)

    p_dec = sub.add_parser(
        "decode-bench",
        help="continuous-batching decode tokens/sec benchmark (PLAN.md §8 M4)",
    )
    p_dec.add_argument("--num-requests", type=int, default=4)
    p_dec.add_argument("--prompt-len", type=int, default=8)
    p_dec.add_argument("--max-new-tokens", type=int, default=16)
    p_dec.add_argument("--vocab-size", type=int, default=128)
    p_dec.add_argument("--hidden-size", type=int, default=64)
    p_dec.add_argument("--intermediate-size", type=int, default=128)
    p_dec.add_argument("--layers", type=int, default=2)
    p_dec.add_argument("--heads", type=int, default=4)
    p_dec.add_argument("--kv-heads", type=int, default=2)
    p_dec.add_argument("--head-dim", type=int, default=8)
    p_dec.add_argument("--dtype", default="float32")
    p_dec.add_argument("--backend", default="cpu")
    p_dec.add_argument("--label", default="llama-toy-decode")
    p_dec.add_argument("--seed", type=int, default=0)
    p_dec.add_argument("--save-baseline", action="store_true")
    p_dec.set_defaults(func=_cmd_decode_bench)

    p_smoke = sub.add_parser(
        "silicon-smoke",
        help="silicon bring-up smoke (PLAN.md §8 M7 exit gate)",
    )
    p_smoke.add_argument("--backend", default="lhsil",
                          help="backend under test (default: lhsil)")
    p_smoke.add_argument("--reference", default="cpu",
                          help="reference backend (default: cpu)")
    p_smoke.add_argument("--seed", type=int, default=0)
    p_smoke.add_argument("--cross-target", action="store_true",
                          help="also run the full cross-target equivalence sweep")
    p_smoke.set_defaults(func=_cmd_silicon_smoke)

    p_full = sub.add_parser(
        "full-sweep",
        help="full performance characterization sweep (PLAN.md §8 M8)",
    )
    p_full.add_argument("--dtype", default="float32")
    p_full.add_argument("--save-baseline", action="store_true",
                         help="overwrite full_sweep_baseline.json")
    p_full.set_defaults(func=_cmd_full_sweep)

    p_reg = sub.add_parser(
        "regression-check",
        help="run every baseline check and gate on regressions (PLAN.md §4.3 / §8 M8)",
    )
    p_reg.add_argument("--regression-factor", type=float, default=1.5)
    p_reg.set_defaults(func=_cmd_regression_check)

    p_ready = sub.add_parser(
        "readiness",
        help="production-inference readiness scorecard (PLAN.md §8 M8 exit gate)",
    )
    p_ready.set_defaults(func=_cmd_readiness)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
