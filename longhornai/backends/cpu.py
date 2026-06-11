"""CPU reference backend (NumPy).

The portable, always-available target. It is the golden-generation and CI
default (PLAN.md §2.2) and the correctness anchor every other backend is
validated against via the cross-target equivalence harness.

Implementation policy:

* As the correctness *anchor*, the CPU backend accumulates in the widest
  practical precision (float64) and rounds only the final result back to the
  I/O dtype. This keeps it as accurate as the float64 golden, so any divergence
  a kernel test reports is genuine, not CPU-accumulation noise.
* The FP16/BF16-in, FP32-accum contract of real tensor cores (PLAN.md §3
  Phase 1) is modeled by the hardware-representative backends (sim / lhsil),
  where accumulation precision is part of what we are validating — not here.
* Outputs preserve the input dtype unless an operator's semantics dictate
  otherwise (e.g. ``embedding_lookup`` with integer ids).

These functions are intentionally simple and readable; they are the reference,
not the performance target.
"""

from __future__ import annotations

import math

import numpy as np

from ..runtime.backend import Backend, register_backend

cpu = Backend("cpu", "Portable NumPy reference backend (golden / CI default).")


def _acc(dtype: np.dtype) -> np.dtype:
    """Accumulation dtype for the reference anchor: float64 for any float input."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return np.dtype(np.float64)
    return dt


# --- GEMM family -------------------------------------------------------------

@cpu.register("gemm")
def gemm(a, b, c=None, alpha=1.0, beta=0.0):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    res = alpha * (a.astype(acc) @ b.astype(acc))
    if c is not None and beta != 0.0:
        res = res + beta * np.asarray(c, dtype=acc)
    return res.astype(out_dt)


@cpu.register("batched_gemm")
def batched_gemm(a, b, alpha=1.0):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    return (alpha * (a.astype(acc) @ b.astype(acc))).astype(out_dt)


@cpu.register("grouped_gemm")
def grouped_gemm(a_list, b_list, alpha=1.0):
    if len(a_list) != len(b_list):
        raise ValueError("grouped_gemm: a_list and b_list length mismatch")
    return [gemm(a, b, alpha=alpha) for a, b in zip(a_list, b_list)]


@cpu.register("tensor_contraction")
def tensor_contraction(a, b, subscripts):
    out_dt = np.result_type(a, b)
    acc = _acc(out_dt)
    return np.einsum(subscripts, a.astype(acc), b.astype(acc)).astype(out_dt)


# --- normalization -----------------------------------------------------------

@cpu.register("layernorm")
def layernorm(x, weight=None, bias=None, eps=1e-5, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    mean = xa.mean(axis=axis, keepdims=True)
    var = xa.var(axis=axis, keepdims=True)
    out = (xa - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight.astype(acc)
    if bias is not None:
        out = out + bias.astype(acc)
    return out.astype(out_dt)


@cpu.register("rmsnorm")
def rmsnorm(x, weight=None, eps=1e-6, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    ms = np.mean(np.square(xa), axis=axis, keepdims=True)
    out = xa / np.sqrt(ms + eps)
    if weight is not None:
        out = out * weight.astype(acc)
    return out.astype(out_dt)


# --- activations -------------------------------------------------------------

@cpu.register("gelu")
def gelu(x, approximate="none"):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        out = 0.5 * xa * (1.0 + np.tanh(c * (xa + 0.044715 * xa**3)))
    else:
        erf = np.vectorize(math.erf, otypes=[np.float64])
        out = 0.5 * xa * (1.0 + erf(xa / math.sqrt(2.0)).astype(acc))
    return out.astype(out_dt)


@cpu.register("silu")
def silu(x):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    return (xa / (1.0 + np.exp(-xa))).astype(out_dt)


# --- elementwise / sequence --------------------------------------------------

@cpu.register("softmax")
def softmax(x, axis=-1):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    m = np.max(xa, axis=axis, keepdims=True)
    e = np.exp(xa - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(out_dt)


@cpu.register("rope")
def rope(x, positions=None, theta=10000.0):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    seq = xa.shape[-2]
    head_dim = xa.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("rope: head_dim must be even")
    half = head_dim // 2
    if positions is None:
        positions = np.arange(seq)
    pos = np.asarray(positions, dtype=acc)
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=acc) * 2.0 / head_dim))
    freqs = np.outer(pos, inv_freq)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    x1 = xa[..., :half]
    x2 = xa[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return np.concatenate([rot1, rot2], axis=-1).astype(out_dt)


@cpu.register("embedding_lookup")
def embedding_lookup(table, ids, scale=1.0):
    out = table[np.asarray(ids)]
    if scale != 1.0:
        out = (out.astype(_acc(out.dtype)) * scale).astype(table.dtype)
    return out


@cpu.register("reduce")
def reduce(x, op="sum", axis=-1, keepdims=False):
    out_dt = x.dtype
    acc = _acc(out_dt)
    xa = x.astype(acc)
    if op == "sum":
        out = np.sum(xa, axis=axis, keepdims=keepdims)
    elif op == "max":
        out = np.max(xa, axis=axis, keepdims=keepdims)
    elif op == "mean":
        out = np.mean(xa, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"unknown reduce op '{op}'")
    return out.astype(out_dt)


register_backend(cpu, default=True)
