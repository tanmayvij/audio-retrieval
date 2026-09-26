import pytest

from evals.metrics import mean_recall, recall_at_k


@pytest.mark.parametrize(("retrieved", "relevant", "k", "expected"), [
    (["a", "b"], ["a", "b"], 3, 1.0),
    (["x", "a", "b"], ["a", "b", "c"], 3, 2 / 3),
    (["x"], ["a"], 5, 0.0),
    ([], ["a"], 5, 0.0),
    (["a", "a", "b"], ["a", "b"], 2, 0.5),
    (["a"], ["a", "b"], 5, 0.5),
    (["x", "a"], ["a"], 1, 0.0),
])
def test_recall(retrieved, relevant, k, expected):
    assert recall_at_k(retrieved, relevant, k) == pytest.approx(expected)


def test_macro_average_weights_queries_equally():
    scores = [recall_at_k(["a"], ["a"], 1), recall_at_k(["b"], ["b", "c", "d"], 1)]
    assert mean_recall(scores) == pytest.approx(2 / 3)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["a"], ["a"], k)


def test_empty_labels_and_empty_average_are_errors():
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k([], [], 5)
    with pytest.raises(ValueError, match="empty"):
        mean_recall([])
