"""Continuous-batching scheduler + paged decode tests (M4 exit gate).

PLAN.md §8 M4 exit gate: "E2E Llama decode under continuous batching" plus
a published tokens/sec baseline. The tests below pin the scheduler's
behavior against the dense (non-paged) reference forward — the same prompt
+ same greedy-argmax sampling must produce identical tokens whether you
run dense forward, paged prefill+decode loop, or the full
:class:`ContinuousBatchingScheduler`.
"""

import numpy as np
import pytest

from longhornai.models import (
    LlamaConfig,
    alloc_paged_kv_state,
    init_random_weights,
    llama_decode_step,
    llama_forward,
    llama_prefill,
)
from longhornai.runtime import (
    ContinuousBatchingScheduler,
    Request,
    SchedulerConfig,
)
from longhornai.runtime import use_backend
from longhornai.validation import assert_close


def _toy_config():
    return LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
    )


# --- prefill / decode parity with the dense forward -----------------------

def test_paged_prefill_matches_dense_forward():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    state = alloc_paged_kv_state(config, batch_size=1, num_blocks=4,
                                 block_size=4, max_blocks_per_seq=2,
                                 dtype=np.float32)
    state.block_table[0, :2] = [0, 1]
    out_paged = llama_prefill(prompt, weights, config, state=state)
    out_dense = llama_forward(prompt, weights, config)
    assert_close(out_paged, out_dense, np.float32, name="prefill↔dense")
    assert state.seq_lens[0] == 5


def test_paged_decode_step_matches_dense_extension():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    state = alloc_paged_kv_state(config, batch_size=1, num_blocks=4,
                                 block_size=4, max_blocks_per_seq=4,
                                 dtype=np.float32)
    state.block_table[0, :4] = [0, 1, 2, 3]
    llama_prefill(prompt, weights, config, state=state)
    next_tok = np.array([7], dtype=np.int64)
    out_decode = llama_decode_step(next_tok, weights, config, state=state)
    extended = np.concatenate([prompt[0], next_tok]).reshape(1, -1)
    out_dense_ext = llama_forward(extended, weights, config)
    assert_close(out_decode[0], out_dense_ext[0, -1], np.float32,
                 name="decode↔dense_ext")
    assert state.seq_lens[0] == 5


# --- scheduler equivalence to single-stream greedy decode -----------------

def _greedy_decode_dense(prompt, weights, config, max_new_tokens):
    seq = prompt.copy()
    out_ids = []
    for _ in range(max_new_tokens):
        logits = llama_forward(seq.reshape(1, -1), weights, config)
        tok = int(np.argmax(logits[0, -1]))
        out_ids.append(tok)
        seq = np.concatenate([seq, np.array([tok], dtype=seq.dtype)])
    return out_ids


def test_scheduler_single_request_matches_greedy_dense():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)

    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=1, num_blocks=8, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32),
    )
    sched.add_request(Request("r1", prompt, max_new_tokens=5))
    sched.run_until_done()
    finished = sched.finished[0]
    expected = _greedy_decode_dense(prompt, weights, config, max_new_tokens=5)
    assert finished.output_ids == expected


def test_scheduler_multi_request_matches_greedy_dense():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompts = [
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([5, 6, 7], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
    ]
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=4, num_blocks=24, block_size=4,
                        max_blocks_per_request=8, cache_dtype=np.float32),
    )
    for i, p in enumerate(prompts):
        sched.add_request(Request(f"r{i}", p, max_new_tokens=4))
    sched.run_until_done()
    by_id = {r.request_id: r for r in sched.finished}
    for i, p in enumerate(prompts):
        expected = _greedy_decode_dense(p, weights, config, max_new_tokens=4)
        assert by_id[f"r{i}"].output_ids == expected, (
            f"r{i}: paged + scheduler diverged from dense greedy"
        )


def test_scheduler_records_iteration_stats():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=2, num_blocks=8, block_size=4,
                        max_blocks_per_request=4, cache_dtype=np.float32),
    )
    sched.add_request(Request("r1", np.array([1, 2, 3, 4], dtype=np.int64), max_new_tokens=3))
    sched.add_request(Request("r2", np.array([5, 6, 7], dtype=np.int64), max_new_tokens=3))
    stats = sched.run_until_done()
    assert stats.iterations >= 2
    assert stats.prefill_tokens == 4 + 3            # both prompts admitted in iter 1
    # Both requests produce 3 output tokens; first comes from prefill sample,
    # the rest are decode samples → 2 decode steps × 2 active = 4.
    assert stats.decode_tokens == 4
    assert stats.requests_completed == 2


# --- scheduler under the FPGA backend (M4 exit gate) ----------------------

def test_scheduler_under_fpga_matches_cpu():
    """E2E continuous batching must produce identical outputs on FPGA."""
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    prompts = [np.array([1, 2, 3, 4], dtype=np.int64),
               np.array([5, 6, 7, 8], dtype=np.int64)]

    def run(backend):
        sched = ContinuousBatchingScheduler(
            weights, config,
            SchedulerConfig(max_batch_size=2, num_blocks=16, block_size=4,
                            max_blocks_per_request=4, cache_dtype=np.float32),
        )
        for i, p in enumerate(prompts):
            sched.add_request(Request(f"r{i}", p, max_new_tokens=3))
        with use_backend(backend):
            sched.run_until_done()
        return {r.request_id: r.output_ids for r in sched.finished}

    cpu_out = run("cpu")
    fpga_out = run("fpga")
    assert cpu_out == fpga_out


# --- pool exhaustion is a clear error, not silent corruption -------------

def test_scheduler_raises_on_block_pool_exhaustion():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    # Pool is too small to hold prompt + max_new_tokens for this request.
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=1, num_blocks=1, block_size=2,
                        max_blocks_per_request=8, cache_dtype=np.float32),
    )
    sched.add_request(Request(
        "r", np.array([1, 2, 3, 4], dtype=np.int64), max_new_tokens=10,
    ))
    with pytest.raises(RuntimeError, match="pool"):
        sched.run_until_done()


def test_scheduler_raises_on_max_blocks_per_request_exceeded():
    config = _toy_config()
    weights = init_random_weights(config, dtype=np.float32, seed=0)
    # Pool has space, but the per-request block cap can't fit the worst case.
    sched = ContinuousBatchingScheduler(
        weights, config,
        SchedulerConfig(max_batch_size=1, num_blocks=64, block_size=2,
                        max_blocks_per_request=2, cache_dtype=np.float32),
    )
    sched.add_request(Request(
        "r", np.array([1, 2, 3, 4], dtype=np.int64), max_new_tokens=10,
    ))
    with pytest.raises(RuntimeError, match="max_blocks_per_request"):
        sched.run_until_done()
