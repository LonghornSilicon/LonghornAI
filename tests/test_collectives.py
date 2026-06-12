"""Collectives kernel tests (M6)."""

import numpy as np
import pytest

import longhornai as lh
from longhornai.validation import assert_close


@pytest.mark.parametrize("op,reduce_fn", [
    ("sum", np.add),
    ("max", np.maximum),
    ("mean", lambda a, b: (a + b) / 2),  # only meaningful for 2 ranks
])
def test_all_reduce_two_rank(rng, op, reduce_fn):
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    out = lh.all_reduce([a, b], op=op)
    expected = reduce_fn(a, b) if op != "mean" else (a + b) / 2
    if op == "max":
        expected = np.maximum(a, b)
    if op == "sum":
        expected = a + b
    assert len(out) == 2
    assert_close(out[0], expected, np.float32, name=f"all_reduce_{op}_rank0")
    assert_close(out[1], expected, np.float32, name=f"all_reduce_{op}_rank1")


def test_all_reduce_n_ranks(rng):
    shards = [rng.standard_normal((4,)).astype(np.float32) for _ in range(4)]
    out = lh.all_reduce(shards, op="sum")
    expected = sum(shards)
    for r, shard in enumerate(out):
        assert_close(shard, expected, np.float32, name=f"all_reduce_4_rank{r}")


def test_all_gather_along_axis_0(rng):
    a = rng.standard_normal((2, 4)).astype(np.float32)
    b = rng.standard_normal((2, 4)).astype(np.float32)
    out = lh.all_gather([a, b], axis=0)
    expected = np.concatenate([a, b], axis=0)
    for r in range(2):
        assert_close(out[r], expected, np.float32, name=f"all_gather_rank{r}")


def test_all_gather_along_axis_1(rng):
    a = rng.standard_normal((4, 2)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    out = lh.all_gather([a, b], axis=1)
    expected = np.concatenate([a, b], axis=1)
    for r in range(2):
        assert_close(out[r], expected, np.float32, name=f"all_gather_axis1_rank{r}")


def test_reduce_scatter(rng):
    a = rng.standard_normal((8, 4)).astype(np.float32)
    b = rng.standard_normal((8, 4)).astype(np.float32)
    out = lh.reduce_scatter([a, b], op="sum", axis=0)
    full = a + b
    chunks = np.split(full, 2, axis=0)
    for r in range(2):
        assert_close(out[r], chunks[r], np.float32, name=f"reduce_scatter_rank{r}")


def test_reduce_scatter_rejects_non_divisible_axis():
    a = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        lh.reduce_scatter([a, a], op="sum", axis=0)


def test_all_to_all_two_rank():
    a = np.array([[10, 11], [12, 13], [14, 15], [16, 17]], dtype=np.float32)
    b = np.array([[20, 21], [22, 23], [24, 25], [26, 27]], dtype=np.float32)
    out = lh.all_to_all([a, b], axis=0)
    # Rank 0 keeps its first half + rank 1's first half.
    expected_r0 = np.array([[10, 11], [12, 13], [20, 21], [22, 23]], dtype=np.float32)
    expected_r1 = np.array([[14, 15], [16, 17], [24, 25], [26, 27]], dtype=np.float32)
    assert_close(out[0], expected_r0, np.float32, name="a2a_rank0")
    assert_close(out[1], expected_r1, np.float32, name="a2a_rank1")


def test_all_to_all_round_trip(rng):
    """Two consecutive all_to_alls should return the original shards."""
    shards = [rng.standard_normal((6, 4)).astype(np.float32) for _ in range(3)]
    once = lh.all_to_all(shards, axis=0)
    twice = lh.all_to_all(once, axis=0)
    for r in range(3):
        assert_close(twice[r], shards[r], np.float32, name=f"a2a_rt_rank{r}")


def test_collectives_cross_target_equivalence(rng):
    """Every collective must agree across every registered backend."""
    from longhornai.validation import assert_cross_target_equivalent
    a = rng.standard_normal((4, 4)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    report = assert_cross_target_equivalent(
        "all_reduce", [a, b], dtype=np.float32, op="sum",
    )
    assert report.passed, report
