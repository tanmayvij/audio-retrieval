import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from evals.faults import FAULTS, run_fault_suite
from evals.gates import gate_mode, make_baseline, validate_baseline
from evals.runner import evaluate, fingerprint, load_queries, run_evaluation, write_json
from evals.unanswerable import analyze_scores, load_unanswerable_queries, semantic_score


ROOT = Path(__file__).resolve().parents[1]
A = "00000000-0000-0000-0000-000000000001"
QUERIES = [{"id": "q", "query": "question", "relevant_chunk_ids": [A]}]
CORPUS = [{"chunk_id": A, "text": "evidence", "has_embedding": True, "filename": "a.mp3"}]
NEGATIVES = [
    {"id": "u1", "query": "absent fact", "category": "near_match", "filename": "a.mp3", "rationale": "Absent."},
    {"id": "u2", "query": "unrelated", "category": "unrelated", "rationale": "Absent topic."},
]


def search(query, mode, *, limit):
    return [{"chunk_id": A, "cosine_similarity": 0.9 if query == "question" else 0.4}]


def baseline():
    return make_baseline(evaluate(QUERIES, CORPUS, search, "model"))


def test_fault_detection_and_matched_controls():
    report = run_fault_suite()
    assert report["passed"]
    assert report["detected_count"] == report["fault_count"] == 6
    assert report["false_alarm_count"] == 0
    assert report["clean_control_count"] == 6
    assert report["detection_rate"] == 1
    assert report["false_alarm_rate"] == 0
    assert {row["fault"] for row in report["cases"]} == set(FAULTS)
    demotion = next(row for row in report["cases"] if row["fault"] == "rank_three_to_four")
    assert demotion["mean_recall"]["semantic"]["5"] == 1.0
    assert "recall_at_5_below_floor" not in demotion["observed_gates"]["semantic"]


def test_unrelated_crash_is_not_counted_as_detection(monkeypatch):
    from evals import faults

    original = faults._evaluate

    def broken(fault, baseline=None):
        if fault == "displaced_semantic_candidates":
            raise TypeError("Unrelated harness bug")
        return original(fault, baseline)

    monkeypatch.setattr(faults, "_evaluate", broken)
    report = faults.run_fault_suite()
    assert not report["passed"]
    assert report["detected_count"] == 5
    assert report["cases"][0]["unexpected_error"]["type"] == "TypeError"


@pytest.mark.parametrize("field", ["embedding_model", "query_set_sha256", "corpus_sha256"])
def test_baseline_identity_mismatch_fails_before_search(field):
    old = baseline()
    old[field] = "different"
    retriever = Mock()
    with pytest.raises(ValueError, match=field):
        evaluate(QUERIES, CORPUS, retriever, "model", baseline=old)
    retriever.assert_not_called()


@pytest.mark.parametrize("bad", [None, {}, {"schema_version": 99}])
def test_invalid_baseline(bad):
    with pytest.raises(ValueError):
        validate_baseline(bad, {}, {"q"})


def test_baseline_rejects_invalid_means_and_query_ids():
    report = evaluate(QUERIES, CORPUS, search, "model")
    old = baseline()
    old["modes"]["semantic"]["mean_recall"]["3"] = 0.9
    with pytest.raises(ValueError, match="mean does not match"):
        validate_baseline(old, report, {"q"})
    old = baseline()
    old["modes"]["semantic"]["queries"][0]["id"] = "other"
    with pytest.raises(ValueError, match="query IDs"):
        validate_baseline(old, report, {"q"})
    old = baseline()
    old["modes"]["semantic"]["queries"][0]["recall"]["3"] = float("nan")
    with pytest.raises(ValueError, match="invalid recall"):
        validate_baseline(old, report, {"q"})


def mode_baseline(scores):
    return {"mean_recall": {"3": sum(scores) / len(scores)},
            "queries": [{"id": str(i), "recall": {"3": value}} for i, value in enumerate(scores)]}


def gate(scores, previous):
    details = [{"id": str(i), "recall": {"3": value}} for i, value in enumerate(scores)]
    return gate_mode({"3": sum(scores) / len(scores), "5": 1.0}, details, mode_baseline(previous))


def test_existing_zero_hit_allowed_but_new_zero_hit_fails_even_when_mean_unchanged():
    assert gate([0, 1, 1, 1, 1], [0, 1, 1, 1, 1])["passed"]
    changed = gate([1, 0, 1, 1, 1], [0, 1, 1, 1, 1])
    assert changed["gate_reasons"] == ["new_zero_hit_queries_at_3"]
    assert changed["baseline_comparison"]["new_zero_hit_queries_at_3"] == ["1"]


def test_partial_regressions_are_visible_when_offset_by_improvements():
    result = gate([1, 0.5, 1, 1, 1], [0.5, 1, 1, 1, 1])
    assert result["passed"]
    assert result["baseline_comparison"]["query_changes"][1]["delta"] == -0.5


def test_strict_baseline_tolerance_and_floor():
    assert gate([0.9 - 5e-13], [0.9])["passed"]
    assert gate([0.9 - 2e-12], [0.9])["gate_reasons"] == ["mean_recall_at_3_regressed"]
    assert gate([0.8], [0.8])["passed"]
    assert "recall_at_3_below_floor" in gate([0.799], [0.799])["gate_reasons"]


def test_committed_baseline_matches_queries_and_is_internally_valid():
    old = json.loads((ROOT / "evals/baseline.json").read_text())
    queries = load_queries(ROOT / "evals/queries.json")
    assert old["query_set_sha256"] == fingerprint(queries)
    validate_baseline(old, old, {row["id"] for row in queries})
    for mode in old["modes"].values():
        assert mode["mean_recall"]["3"] == 0.825
        assert [q["id"] for q in mode["queries"] if q["recall"]["3"] == 0] == ["q13"]


def test_score_sweep_ties_boundaries_and_counts():
    positives = [{"id": "p1", "score": 0.5}, {"id": "p2", "score": 0.8}]
    negatives = [{"id": "u1", "score": 0.2, "category": "unrelated"},
                 {"id": "u2", "score": 0.5, "category": "near_match"}]
    report = analyze_scores(positives, negatives)
    sweep = report["threshold_sweep"]
    assert len(sweep) == 5
    assert sweep[0]["false_alarm_rate"] == sweep[0]["unanswerable"]["all"]["detection_rate"] == 0
    assert sweep[-1]["false_alarm_rate"] == sweep[-1]["unanswerable"]["all"]["detection_rate"] == 1
    middle = next(row for row in sweep if row["threshold"] == 0.5)
    assert middle["false_alarms"] == 0
    assert middle["unanswerable"]["all"] == {"detected": 1, "count": 2, "detection_rate": 0.5}
    assert middle["unanswerable"]["near_match"]["detected"] == 0
    assert report["selected_threshold"] is None


@pytest.mark.parametrize("results", [[], [{}]] + [
    [{"cosine_similarity": value}] for value in [None, True, "0.5", float("nan"), float("inf"), 1.1]
])
def test_invalid_semantic_scores_are_errors(results):
    with pytest.raises(ValueError):
        semantic_score(results, "q")


def test_highest_semantic_score():
    assert semantic_score([{"cosine_similarity": 0.4}, {"cosine_similarity": 0.8}], "q") == 0.8


@pytest.mark.parametrize("change", [
    {"id": "q"}, {"query": " "}, {"rationale": ""}, {"category": "unknown"},
    {"filename": ""}, {"relevant_chunk_ids": [A]},
])
def test_invalid_negative_labels(tmp_path, change):
    rows = deepcopy(NEGATIVES)
    rows[0].update(change)
    path = tmp_path / "negative.json"
    write_json(path, rows)
    with pytest.raises(ValueError):
        load_unanswerable_queries(path, {"q"})


def test_committed_negative_set():
    rows = load_unanswerable_queries(ROOT / "evals/unanswerable_queries.json", {q["id"] for q in QUERIES})
    assert len(rows) == 12
    near = [row for row in rows if row["category"] == "near_match"]
    assert len(near) == 6
    assert {row["filename"] for row in near} == {f"output_{i}.mp3" for i in range(1, 7)}


def test_negatives_do_not_change_recall_or_select_cutoff():
    report = evaluate(QUERIES, CORPUS, search, "model", baseline=baseline(), unanswerable=NEGATIVES)
    assert report["passed"]
    assert report["query_count"] == 1
    assert report["modes"]["semantic"]["mean_recall"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    assert len(report["unanswerable"]["unanswerable"]["queries"]) == 2
    assert report["unanswerable"]["selected_threshold"] is None


def evaluation_paths(tmp_path):
    paths = {key: tmp_path / name for key, name in [
        ("queries_path", "queries.json"), ("baseline_path", "baseline.json"),
        ("unanswerable_path", "negatives.json"), ("report_path", "report.json")
    ]}
    write_json(paths["queries_path"], QUERIES)
    write_json(paths["baseline_path"], baseline())
    write_json(paths["unanswerable_path"], NEGATIVES)
    write_json(paths["report_path"], {"status": "success", "passed": True, "created_at": "old"})
    return paths


@pytest.mark.parametrize("failure", ["database", "baseline", "labels", "retrieval", "scores", "filename"])
def test_fresh_error_report_replaces_stale_success(tmp_path, failure):
    paths = evaluation_paths(tmp_path)
    loader = Mock(return_value=CORPUS)
    retriever = search
    if failure == "database":
        loader.side_effect = RuntimeError("Database unavailable")
    elif failure == "baseline":
        paths["baseline_path"].write_text("{}")
    elif failure == "labels":
        paths["queries_path"].write_text("[]")
    elif failure == "retrieval":
        retriever = Mock(side_effect=OSError("offline"))
    elif failure == "scores":
        retriever = Mock(return_value=[])
    else:
        loader.return_value = [{**CORPUS[0], "filename": "other.mp3"}]
    with pytest.raises((ValueError, RuntimeError)):
        run_evaluation(**paths, corpus_loader=loader, search=retriever, model="model")
    report = json.loads(paths["report_path"].read_text())
    assert report["status"] == "error"
    assert report["passed"] is False
    assert report["created_at"] != "old"
    assert "error" in report


def test_run_persists_success_and_does_not_change_baseline(tmp_path):
    paths = evaluation_paths(tmp_path)
    original = paths["baseline_path"].read_bytes()
    report = run_evaluation(**paths, corpus_loader=lambda: CORPUS, search=search, model="model")
    assert report["status"] == "success"
    assert json.loads(paths["report_path"].read_text()) == report
    assert paths["baseline_path"].read_bytes() == original


def test_gate_failure_is_saved_before_assertion(tmp_path):
    paths = evaluation_paths(tmp_path)
    retriever = Mock(return_value=[{"chunk_id": "irrelevant", "cosine_similarity": 0.3}])
    report = run_evaluation(**paths, corpus_loader=lambda: CORPUS, search=retriever, model="model")
    assert report["status"] == "failed"
    assert not json.loads(paths["report_path"].read_text())["passed"]


def test_report_cannot_overwrite_baseline(tmp_path):
    paths = evaluation_paths(tmp_path)
    original = paths["baseline_path"].read_bytes()
    paths["report_path"] = paths["baseline_path"]
    with pytest.raises(ValueError, match="overwrite"):
        run_evaluation(**paths, corpus_loader=lambda: CORPUS, search=search, model="model")
    assert paths["baseline_path"].read_bytes() == original
