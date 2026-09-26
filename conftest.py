"""Opt in to database/model evaluations explicitly."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-evals", action="store_true", help="Run live retrieval evaluations")
    parser.addoption("--eval-queries", default="evals/queries.json", help="Labeled query JSON file")
    parser.addoption("--eval-report", default="evals/reports/recall.json", help="JSON report output")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-evals"):
        return
    skip = pytest.mark.skip(reason="Use --run-evals to run database/model evaluations")
    for item in items:
        if item.get_closest_marker("eval"):
            item.add_marker(skip)
