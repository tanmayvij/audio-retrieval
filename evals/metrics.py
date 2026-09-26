"""Pure Recall@k calculations; no database or model dependencies."""

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValueError("Recall requires at least one relevant chunk.")
    return len(set(retrieved[:k]) & relevant_ids) / len(relevant_ids)


def mean_recall(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("Cannot average an empty query set.")
    return sum(scores) / len(scores)
