"""Run with python -m pytest evals/ --run-evals -m eval -s."""

from pathlib import Path

import pytest

from evals.runner import format_summary, load_corpus, run_evaluation


pytestmark = pytest.mark.eval


@pytest.fixture(scope="session")
def recall_report(pytestconfig):
    from app.config import EMBEDDING_MODEL

    def search(*args, **kwargs):
        from app.services.retrieval import search as live_search

        return live_search(*args, **kwargs)

    path = Path(pytestconfig.getoption("--eval-report"))
    report = run_evaluation(
        queries_path=Path(pytestconfig.getoption("--eval-queries")),
        baseline_path=Path(pytestconfig.getoption("--eval-baseline")),
        unanswerable_path=Path(pytestconfig.getoption("--eval-unanswerable-queries")),
        report_path=path, corpus_loader=load_corpus, search=search, model=EMBEDDING_MODEL,
    )
    print("\n" + format_summary(report))
    print(f"Per-query report: {path}")
    return report


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_retrieval_gates(recall_report, mode):
    result = recall_report["modes"][mode]
    assert result["passed"], (
        f"{mode} failed gates: {', '.join(result['gate_reasons'])}. "
        "Inspect the JSON report for missed relevant chunks."
    )
