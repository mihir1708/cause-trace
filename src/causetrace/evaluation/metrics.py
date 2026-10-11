"""Retrieval metrics (docs/EVAL_PROTOCOL.md). Ranked inputs are event ids; repeated events
count once, at their first position, and only the top ``k`` distinct events are scored."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

K = 10
RELEVANCE_THRESHOLD = 0.5
EMPTY: frozenset[str] = frozenset()


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|A and B| / |A or B| (set intersection over union); 0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def distinct_top_k(ranked: Sequence[str], k: int = K) -> list[str]:
    """The first ``k`` distinct ids, in order."""
    seen: list[str] = []
    for item in ranked:
        if item not in seen:
            seen.append(item)
            if len(seen) == k:
                break
    return seen


def _label_sets(
    ranked: Sequence[str], labels: Mapping[str, frozenset[str]], k: int
) -> list[frozenset[str]]:
    return [labels.get(ev, EMPTY) for ev in distinct_top_k(ranked, k)]


def label_recall_at_k(
    gold: frozenset[str], ranked: Sequence[str], labels: Mapping[str, frozenset[str]], k: int = K
) -> float:
    """|G intersect (L_1 union ... union L_k)| / |G|."""
    if not gold:
        raise ValueError("label recall is undefined for an empty gold set")
    covered: set[str] = set()
    for ls in _label_sets(ranked, labels, k):
        covered |= ls
    return len(gold & covered) / len(gold)


def dcg(gains: Sequence[float]) -> float:
    """Σ g_i / log2(i + 1), ranks from 1."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ideal_gains(
    gold: frozenset[str], pool: Counter[frozenset[str]] | Iterable[frozenset[str]], k: int = K
) -> list[float]:
    """The k highest J(G, ·) over the candidate pool (label sets with multiplicity)."""
    counts = pool if isinstance(pool, Counter) else Counter(pool)
    gains: list[float] = []
    for j, n in sorted(((jaccard(gold, ls), n) for ls, n in counts.items()), reverse=True):
        if j <= 0 or len(gains) >= k:
            break
        gains.extend([j] * min(n, k - len(gains)))
    return gains


def ndcg_at_k(
    gold: frozenset[str],
    ranked: Sequence[str],
    labels: Mapping[str, frozenset[str]],
    pool: Counter[frozenset[str]] | Iterable[frozenset[str]],
    k: int = K,
) -> float:
    """DCG@k with Jaccard gain over the ideal DCG@k from ``pool``; 0 when the ideal is 0."""
    idcg = dcg(ideal_gains(gold, pool, k))
    if idcg == 0:
        return 0.0
    return dcg([jaccard(gold, ls) for ls in _label_sets(ranked, labels, k)]) / idcg


def first_relevant_rank(
    gold: frozenset[str],
    ranked: Sequence[str],
    labels: Mapping[str, frozenset[str]],
    k: int = K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> int | None:
    """1-based rank of the first event with J(G, L) >= threshold in the top k, else None."""
    for rank, ls in enumerate(_label_sets(ranked, labels, k), start=1):
        if jaccard(gold, ls) >= threshold:
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def known_item_rank(source: str, ranked: Sequence[str], k: int = K) -> int | None:
    """1-based rank of ``source`` among the top k distinct events, else None."""
    top = distinct_top_k(ranked, k)
    return top.index(source) + 1 if source in top else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of values <= it."""
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1]
