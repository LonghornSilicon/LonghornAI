"""LonghornAI runtime: backend abstraction and kernel dispatch.

The runtime is the thin layer that maps a logical operator name (``"gemm"``,
``"flash_attention_v2"``, ...) to a concrete, target-specialized implementation.
Per PLAN.md §2.2 the same operator is authored once and lowered to many targets;
the dispatch registry is where that selection happens at call time.

Today only the CPU (NumPy) reference backend is registered. Additional backends
(sim, rtl, fpga, lhsil) register themselves the same way and become selectable
via :func:`set_default_backend` / :func:`use_backend` without touching kernels.
"""

from __future__ import annotations

from .backend import (
    Backend,
    available_backends,
    dispatch,
    get_backend,
    register_backend,
    set_default_backend,
    use_backend,
)

# Importing the cpu backend registers it as the default target.
from ..backends import cpu as _cpu  # noqa: F401
# Pre-silicon shims (M3/M4/M7) — registered after CPU so CPU stays the default.
from ..backends import sim as _sim  # noqa: F401
from ..backends import rtl as _rtl  # noqa: F401
from ..backends import fpga as _fpga  # noqa: F401
from ..backends import lhsil as _lhsil  # noqa: F401

from .scheduler import (
    ContinuousBatchingScheduler,
    PrefixCache,
    Request,
    RequestState,
    SchedulerConfig,
    SchedulerStats,
)
from .speculative import (
    SpeculativeDecoder,
    SpeculativeStats,
    greedy_target_decode,
)
from .tensor_parallel import (
    TPLayerWeights,
    TPWeights,
    llama_forward_tp,
    shard_column,
    shard_llama_for_tp,
    shard_row,
)
from .expert_parallel import (
    EPLayerWeights,
    EPWeights,
    mixtral_forward_ep,
    shard_mixtral_for_ep,
)
from .pipeline_parallel import (
    PPStage,
    PPWeights,
    llama_forward_pp,
    micro_batch_forward_pp,
    shard_llama_for_pp,
)
from .inference import (
    GenerationResult,
    LonghornInference,
    inference_session,
)

__all__ = [
    "Backend",
    "dispatch",
    "get_backend",
    "register_backend",
    "set_default_backend",
    "available_backends",
    "use_backend",
    # M4 scheduler
    "ContinuousBatchingScheduler",
    "Request",
    "RequestState",
    "SchedulerConfig",
    "SchedulerStats",
    # M5 prefix cache
    "PrefixCache",
    # M5 speculative decoding
    "SpeculativeDecoder",
    "SpeculativeStats",
    "greedy_target_decode",
    # M6 tensor parallelism
    "TPLayerWeights",
    "TPWeights",
    "shard_column",
    "shard_row",
    "shard_llama_for_tp",
    "llama_forward_tp",
    # M6 expert parallelism
    "EPLayerWeights",
    "EPWeights",
    "shard_mixtral_for_ep",
    "mixtral_forward_ep",
    # M7 pipeline parallelism
    "PPStage",
    "PPWeights",
    "shard_llama_for_pp",
    "llama_forward_pp",
    "micro_batch_forward_pp",
    # M8 production runtime
    "GenerationResult",
    "LonghornInference",
    "inference_session",
]
