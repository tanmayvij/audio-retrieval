import json
from unittest.mock import Mock

import pytest

from evals.runner import evaluate, format_summary, load_queries, validate_labels, write_json


A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"
C = "00000000-0000-0000-0000-000000000003"


def labeled_query():
    return {"id": "q01", "query": "What is discussed?", "relevant_chunk_ids": [A, B]}


def corpus():
    return [{"chunk_id": chunk_id, "text": chunk_id, "has_embedding": True} for chunk_id in [A, B, C]]


def test_load_valid_queries(tmp_path):
    path = tmp_path / "queries.json"
    write_json(path, [labeled_query()])
    assert load_queries(path) == [labeled_query()]


@pytest.mark.parametrize("payload", [
    [], {}, ["invalid"],
    [labeled_query(), labeled_query()],
    [{**labeled_query(), "id": " "}],
    [{**labeled_query(), "query": " "}],
    [{**labeled_query(), "relevant_chunk_ids": []}],
    [{**labeled_query(), "relevant_chunk_ids": [A, A]}],
    [{**labeled_query(), "relevant_chunk_ids": ["not-a-uuid"]}],
    [{**labeled_query(), "relevant_chunk_ids": [42]}],
])
def test_reject_invalid_labels(tmp_path, payload):
    path = tmp_path / "queries.json"
    write_json(path, payload)
    with pytest.raises(ValueError):
        load_queries(path)


def test_reject_stale_or_unembedded_labels():
    with pytest.raises(ValueError, match="missing"):
        validate_labels([labeled_query()], corpus()[:1])
    rows = corpus()
    rows[0]["has_embedding"] = False
    with pytest.raises(ValueError, match="no embedding"):
        validate_labels([labeled_query()], rows)


def test_evaluation_reports_each_mode_and_all_prefixes(tmp_path):
    search = Mock(side_effect=[
        [{"chunk_id": A}, {"chunk_id": C}],
        [{"chunk_id": A}, {"chunk_id": C}, {"chunk_id": B}],
    ])
    report = evaluate([labeled_query()], corpus(), search, "test-model")
    assert search.call_count == 2
    search.assert_any_call("What is discussed?", "semantic", limit=5)
    search.assert_any_call("What is discussed?", "hybrid", limit=5)
    assert report["modes"]["semantic"]["mean_recall"] == {"1": 0.5, "3": 0.5, "5": 0.5}
    assert report["modes"]["hybrid"]["mean_recall"] == {"1": 0.5, "3": 1.0, "5": 1.0}
    assert report["modes"]["semantic"]["queries"][0]["missed_relevant_chunk_ids"]["5"] == [B]
    assert not report["modes"]["semantic"]["passed"]
    assert report["modes"]["hybrid"]["passed"]
    assert not report["passed"]
    assert "Recall@1" in format_summary(report)
    assert "FAIL" in format_summary(report)
    path = tmp_path / "reports" / "recall.json"
    write_json(path, report)
    assert json.loads(path.read_text()) == report


def test_retrieval_failure_is_not_scored_as_zero():
    with pytest.raises(RuntimeError, match="semantic, query q01"):
        evaluate([labeled_query()], corpus(), Mock(side_effect=OSError("offline")), "test-model")


def test_fingerprints_track_corpus_and_queries():
    search = Mock(return_value=[])
    original = evaluate([labeled_query()], corpus(), search, "test-model")
    reordered = evaluate([labeled_query()], list(reversed(corpus())), search, "test-model")
    changed_rows = corpus()
    changed_rows[0]["text"] = "changed transcript"
    changed = evaluate([labeled_query()], changed_rows, search, "test-model")
    changed_query = evaluate([{**labeled_query(), "query": "Other question?"}], corpus(), search, "test-model")
    assert original["corpus_sha256"] == reordered["corpus_sha256"]
    assert original["corpus_sha256"] != changed["corpus_sha256"]
    assert original["query_set_sha256"] != changed_query["query_set_sha256"]


def test_threshold_includes_exactly_eighty_percent():
    ids = [str(index) for index in range(5)]
    queries = [{"id": "q", "query": "question", "relevant_chunk_ids": ids}]
    rows = [{"chunk_id": item, "text": item, "has_embedding": True} for item in ids]
    report = evaluate(queries, rows, Mock(return_value=[{"chunk_id": item} for item in ids[:4]]), "test-model")
    assert report["passed"]
