"""Run with python -m pytest evals/ --run-evals -m eval -s."""

from pathlib import Path

import pytest

from evals.runner import evaluate, format_summary, load_corpus, load_queries, write_json


pytestmark = pytest.mark.eval


@pytest.fixture(scope="session")
def recall_report(pytestconfig):
    from app.config import EMBEDDING_MODEL
    from app.services.retrieval import search

    queries = load_queries(Path(pytestconfig.getoption("--eval-queries")))
    report = evaluate(queries, load_corpus(), search, EMBEDDING_MODEL)
    path = Path(pytestconfig.getoption("--eval-report"))
    write_json(path, report)
    print("\n" + format_summary(report))
    print(f"Per-query report: {path}")
    return report


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_mean_recall_at_five(recall_report, mode):
    result = recall_report["modes"][mode]
    assert result["passed"], (
        f"{mode} mean Recall@5 = {result['mean_recall']['5']:.3f}; required >= 0.80. "
        "Inspect the JSON report for missed relevant chunks."
    )
