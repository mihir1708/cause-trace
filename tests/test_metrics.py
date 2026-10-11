"""Metric functions against small hand-computed examples."""

import math
from collections import Counter

import pytest

from causetrace.evaluation import metrics as m

A, B, C, D = "01-01-01", "02-02-02", "03-03-03", "04-04-04"
G = frozenset({A, B, C})
LABELS = {
    "e1": frozenset({A}),  # J(G, e1) = 1/3
    "e2": frozenset({B, D}),  # J(G, e2) = 1/4
    "e3": frozenset({A, B, C}),  # J(G, e3) = 1
    "e4": frozenset(),  # no labels: J = 0
}
POOL = list(LABELS.values())


def test_jaccard() -> None:
    assert m.jaccard(G, LABELS["e1"]) == pytest.approx(1 / 3)
    assert m.jaccard(G, LABELS["e2"]) == pytest.approx(1 / 4)
    assert m.jaccard(G, G) == 1.0
    assert m.jaccard(frozenset(), frozenset()) == 0.0


def test_distinct_top_k() -> None:
    assert m.distinct_top_k(["a", "a", "b", "a", "c", "d"], 3) == ["a", "b", "c"]
    assert m.distinct_top_k([], 10) == []


def test_label_recall() -> None:
    assert m.label_recall_at_k(G, ["e1", "e2"], LABELS) == pytest.approx(2 / 3)
    assert m.label_recall_at_k(G, ["e3"], LABELS) == 1.0
    assert m.label_recall_at_k(G, [], LABELS) == 0.0  # empty top 10
    assert m.label_recall_at_k(G, ["e4", "unknown"], LABELS) == 0.0
    # A duplicate counts once: the top 2 distinct events are e1 and e2, not e1 twice.
    assert m.label_recall_at_k(G, ["e1", "e1", "e2"], LABELS, k=2) == pytest.approx(2 / 3)
    # Only the top k count.
    assert m.label_recall_at_k(G, ["e1", "e3"], LABELS, k=1) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        m.label_recall_at_k(frozenset(), ["e1"], LABELS)


def test_dcg() -> None:
    assert m.dcg([1.0, 1.0, 1.0]) == pytest.approx(1 + 1 / math.log2(3) + 1 / 2)
    assert m.dcg([]) == 0.0


def test_ideal_gains_with_multiplicity() -> None:
    assert m.ideal_gains(G, POOL) == pytest.approx([1.0, 1 / 3, 1 / 4])
    pool = Counter({LABELS["e3"]: 2, LABELS["e2"]: 20})
    assert m.ideal_gains(G, pool, k=3) == pytest.approx([1.0, 1.0, 1 / 4])


def test_ndcg() -> None:
    ideal = 1 + (1 / 3) / math.log2(3) + (1 / 4) / 2
    got = m.ndcg_at_k(G, ["e1", "e2", "e3"], LABELS, POOL)
    assert got == pytest.approx((1 / 3 + (1 / 4) / math.log2(3) + 1 / 2) / ideal)
    assert m.ndcg_at_k(G, ["e3", "e1", "e2"], LABELS, POOL) == pytest.approx(1.0)
    assert m.ndcg_at_k(G, [], LABELS, POOL) == 0.0  # empty top 10
    # Duplicates count once: e3, e3, e1 scores like e3, e1.
    assert m.ndcg_at_k(G, ["e3", "e3", "e1"], LABELS, POOL) == pytest.approx(
        (1 + (1 / 3) / math.log2(3)) / ideal
    )
    # No pool event overlaps G: the ideal DCG is 0, so nDCG is 0.
    assert m.ndcg_at_k(G, ["e4"], LABELS, [frozenset({D})]) == 0.0


def test_first_relevant_rank_mrr_hit() -> None:
    assert m.first_relevant_rank(G, ["e1", "e2", "e3"], LABELS) == 3
    assert m.reciprocal_rank(3) == pytest.approx(1 / 3)
    assert m.first_relevant_rank(G, ["e1", "e2"], LABELS) is None  # nothing reaches J >= 0.5
    assert m.reciprocal_rank(None) == 0.0
    assert m.first_relevant_rank(G, [], LABELS) is None  # empty top 10
    assert m.first_relevant_rank(G, ["e1", "e1", "e3"], LABELS) == 2  # duplicate collapsed
    assert m.first_relevant_rank(G, ["e1", "e2", "e3"], LABELS, k=2) is None
    # The threshold is inclusive: J = 1/2 is relevant.
    assert m.first_relevant_rank(frozenset({A, B}), ["e1"], LABELS) == 1


def test_known_item_rank() -> None:
    assert m.known_item_rank("x", ["a", "a", "x"]) == 2
    assert m.known_item_rank("x", ["a", "x"], k=1) is None
    assert m.known_item_rank("x", []) is None
    assert m.known_item_rank("x", [f"e{i}" for i in range(10)] + ["x"]) is None  # rank 11


def test_mean_and_percentile() -> None:
    assert m.mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert m.mean([]) is None
    values = [float(v) for v in range(1, 11)]  # 1..10
    assert m.percentile(values, 50) == 5.0
    assert m.percentile(values, 95) == 10.0
    assert m.percentile(list(reversed(values)), 10) == 1.0
    assert m.percentile([7.0], 95) == 7.0
    assert m.percentile([], 50) is None
    with pytest.raises(ValueError):
        m.percentile(values, 0)
