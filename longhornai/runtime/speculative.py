"""Speculative decoding (PLAN.md §3 Phase 3 / §8 M5).

Draft-then-verify: a small **draft** model generates ``k`` candidate tokens
sequentially; the **target** (full) model verifies them in one batched
forward, accepting tokens up to (and including) the first position where
the draft and target disagree. With greedy sampling the acceptance test
is exact: target's argmax must match the draft token.

Per accepted iteration we get up to ``k + 1`` new tokens for the cost of
one target forward — a 2-3× walltime speedup on real hardware when the
draft model is well-aligned. Lossless under greedy decoding (the output
distribution matches running the target alone).

Public surface:

* :class:`SpeculativeDecoder` — wires a draft and target model together
  and exposes a :meth:`generate` method that produces a single greedy
  continuation. The reference uses dense :func:`llama_forward` for both
  models — easy to reason about and validate. The scheduler hook for
  batched speculative decoding lives alongside in this module.
* :class:`SpeculativeStats` — tokens proposed / accepted / acceptance
  rate, exposed by the scheduler benchmarks.

PLAN.md §8 M5 deliverable: speculative decoding + lossless agreement
with greedy single-stream decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..models.llama import (
    LlamaConfig,
    LlamaWeights,
    llama_forward,
)


@dataclass
class SpeculativeStats:
    """Aggregate accept/propose counts for a speculative-decoding run."""

    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


@dataclass
class SpeculativeDecoder:
    """Draft-and-verify wrapper over a target + draft Llama pair.

    Both models must share the same tokenizer / vocab — the draft is a
    smaller model trained on the same distribution. The decode loop is
    greedy and deterministic, so the produced sequence is identical to
    running the target alone (lossless guarantee).
    """

    target_weights: LlamaWeights
    target_config: LlamaConfig
    draft_weights: LlamaWeights
    draft_config: LlamaConfig
    speculate_k: int = 4

    def __post_init__(self) -> None:
        if self.target_config.vocab_size != self.draft_config.vocab_size:
            raise ValueError(
                "target and draft must share vocab_size; got "
                f"{self.target_config.vocab_size} vs {self.draft_config.vocab_size}"
            )

    # --- single-stream generation ------------------------------------------

    def generate(
        self,
        prompt_ids: np.ndarray,
        *,
        max_new_tokens: int,
        stats: "SpeculativeStats | None" = None,
    ) -> List[int]:
        """Greedy generation with draft-and-verify. Returns the new tokens."""
        if stats is None:
            stats = SpeculativeStats()
        seq = prompt_ids.copy()
        out: List[int] = []

        while len(out) < max_new_tokens:
            need = max_new_tokens - len(out)
            k = min(self.speculate_k, need)

            # 1) Draft proposes k tokens by greedy autoregressive decoding.
            draft_seq = seq.copy()
            proposed: List[int] = []
            for _ in range(k):
                logits = llama_forward(
                    draft_seq.reshape(1, -1), self.draft_weights, self.draft_config,
                )
                tok = int(np.argmax(logits[0, -1]))
                proposed.append(tok)
                draft_seq = np.concatenate(
                    [draft_seq, np.array([tok], dtype=draft_seq.dtype)]
                )
            stats.proposed += len(proposed)

            # 2) Target verifies in one batched forward over (seq + proposals).
            verify_seq = np.concatenate(
                [seq, np.array(proposed, dtype=seq.dtype)]
            ).reshape(1, -1)
            target_logits = llama_forward(
                verify_seq, self.target_weights, self.target_config,
            )
            stats.target_calls += 1

            # The target's logits at position ``len(seq) - 1 + j`` are the
            # next-token distribution conditioned on the prefix + first j
            # proposals. Accept proposals while target argmax matches.
            n_accepted = 0
            target_next_after_prefix = len(seq) - 1
            for j, prop in enumerate(proposed):
                t_logits = target_logits[0, target_next_after_prefix + j]
                t_argmax = int(np.argmax(t_logits))
                if t_argmax == prop:
                    n_accepted += 1
                else:
                    break

            # Accept the matched prefix.
            stats.accepted += n_accepted
            for j in range(n_accepted):
                out.append(proposed[j])
                seq = np.concatenate(
                    [seq, np.array([proposed[j]], dtype=seq.dtype)]
                )
                if len(out) >= max_new_tokens:
                    return out

            # 3) Always emit one more "bonus" token from the target's logits
            # at the first-mismatch position — that token is from the target
            # distribution and is therefore lossless to accept regardless.
            bonus_idx = target_next_after_prefix + n_accepted
            bonus = int(np.argmax(target_logits[0, bonus_idx]))
            out.append(bonus)
            seq = np.concatenate(
                [seq, np.array([bonus], dtype=seq.dtype)]
            )

        return out


def greedy_target_decode(
    prompt_ids: np.ndarray,
    weights: LlamaWeights,
    config: LlamaConfig,
    *,
    max_new_tokens: int,
) -> List[int]:
    """Reference: greedy autoregressive decode using only the target model."""
    seq = prompt_ids.copy()
    out: List[int] = []
    for _ in range(max_new_tokens):
        logits = llama_forward(seq.reshape(1, -1), weights, config)
        tok = int(np.argmax(logits[0, -1]))
        out.append(tok)
        seq = np.concatenate([seq, np.array([tok], dtype=seq.dtype)])
    return out


__all__ = [
    "SpeculativeDecoder",
    "SpeculativeStats",
    "greedy_target_decode",
]
