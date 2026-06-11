"""``lhai`` CLI entry point.

Subcommands:

    lhai backends                 List registered execution backends.
    lhai kernels                  List Phase-1 kernels and their goldens.
    lhai validate [--dtype ...]   Differential-test every kernel vs its golden.
    lhai bench gemm [-m -k -n]    Benchmark GEMM and print roofline metrics.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .. import kernels, runtime
from ..benchmarks import benchmark, cost_for
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
