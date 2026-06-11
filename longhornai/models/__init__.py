"""End-to-end model graphs assembled from LonghornAI kernels (PLAN.md §7).

M1 ships a single reference block — a Llama-style pre-norm transformer MLP +
RMSNorm path — composed entirely from Phase-1 kernels, to prove that kernel
*composition* is correct, not just individual ops. Full attention blocks and
the model support matrix land with Phase-2/3 milestones.
"""

from __future__ import annotations

from .llama_block import llama_mlp_block

__all__ = ["llama_mlp_block"]
