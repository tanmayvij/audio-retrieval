"""Deterministic synthetic detector checks; no database or ML model is used.

Run ``python -m evals.faults`` to save an inspectable coverage report.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from evals.gates import make_baseline
from evals.runner import evaluate, write_json


FAULTS = {
    "displaced_semantic_candidates": ("semantic", "mean_recall_at_3_regressed"),
    "removed_recording_candidates": ("hybrid", "new_zero_hit_queries_at_3"),
    "disabled_lexical_branch": ("hybrid", "new_zero_hit_queries_at_3"),
    "rank_three_to_four": ("semantic", "mean_recall_at_3_regressed"),
    "missing_labeled_embedding": ("ValueError", "has no embedding"),
    "retrieval_exception": ("RuntimeError", "Retrieval failed for semantic, query f01"),
}


def _fixture() -> tuple[list[dict], list[dict]]:
    corpus = [
        {"chunk_id": str(UUID(int=i + 1)), "text": f"Synthetic evidence {i + 1}",
         "has_embedding": True, "recording_id": "recording-b" if i == 1 else "recording-a"}
        for i in range(9)
    ]
    queries = [{"id": f"f{i + 1:02}", "query": f"synthetic question {i}",
                "relevant_chunk_ids": [corpus[i]["chunk_id"]]} for i in range(5)]
    return queries, corpus


def _evaluate(fault: str | None, baseline: dict | None = None) -> dict:
    # Replace only IO boundaries; search(), rank_candidates(), and evaluate() are real.
    from app.services import retrieval

    queries, corpus = _fixture()
    if fault == "missing_labeled_embedding":
        corpus[0]["has_embedding"] = False

    def semantic_candidates(connection, vector, limit):
        if fault == "retrieval_exception":
            raise OSError("Injected retrieval outage")
        index = int(vector.rsplit(" ", 1)[1])
        relevant = corpus[index]
        distractors = corpus[5:]
        # f01 is at rank 3; f05 needs lexical evidence to enter the hybrid top 3.
        position = 2 if index == 0 else 3 if index == 4 else 0
        ordered = [*distractors[:position], relevant, *distractors[position:]]
        if fault == "displaced_semantic_candidates" and index == 0:
            ordered = [*distractors, corpus[1], relevant]
        if fault == "rank_three_to_four" and index == 0:
            ordered[2], ordered[3] = ordered[3], ordered[2]
        if fault == "removed_recording_candidates":
            ordered = [row for row in ordered if row["recording_id"] != "recording-b"]
        return [{**row, "cosine_similarity": 0.95 - i * 0.05} for i, row in enumerate(ordered[:limit])]

    def lexical_candidates(connection, query, limit):
        if fault == "disabled_lexical_branch":
            return []
        return [deepcopy(corpus[4])] if query == queries[4]["query"] else []

    with (patch.object(retrieval, "connect", side_effect=lambda: nullcontext(object())),
          patch.object(retrieval, "embed_query", side_effect=lambda query: query),
          patch.object(retrieval, "semantic_candidates", side_effect=semantic_candidates),
          patch.object(retrieval, "lexical_candidates", side_effect=lexical_candidates)):
        return evaluate(queries, corpus, retrieval.search, "synthetic-model", baseline=baseline)


def run_fault_suite() -> dict:
    baseline = make_baseline(_evaluate(None))
    cases = []
    for fault, (target, expected) in FAULTS.items():
        control = _evaluate(None, baseline)
        case = {"fault": fault, "expected_signal": {"target": target, "signal": expected},
                "clean_control_passed": control["passed"], "detected": False,
                "unexpected_error": None}
        try:
            report = _evaluate(fault, baseline)
            case["observed_gates"] = {mode: row["gate_reasons"] for mode, row in report["modes"].items()}
            case["mean_recall"] = {mode: row["mean_recall"] for mode, row in report["modes"].items()}
            case["detected"] = expected in case["observed_gates"].get(target, [])
        except Exception as error:
            case["observed_error"] = {"type": type(error).__name__, "message": str(error)}
            case["detected"] = type(error).__name__ == target and expected in str(error)
            if case["detected"] and fault == "retrieval_exception":
                case["detected"] = isinstance(error.__cause__, OSError) and str(error.__cause__) == "Injected retrieval outage"
            if not case["detected"]:
                case["unexpected_error"] = case["observed_error"]
        cases.append(case)
    detected = sum(row["detected"] for row in cases)
    false_alarms = sum(not row["clean_control_passed"] for row in cases)
    passed = detected == len(cases) and false_alarms == 0
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "success" if passed else "failed", "passed": passed,
        "coverage": "Synthetic detector coverage only; not a production detection-rate estimate.",
        "fault_count": len(cases), "detected_count": detected, "missed_count": len(cases) - detected,
        "detection_rate": detected / len(cases), "clean_control_count": len(cases),
        "false_alarm_count": false_alarms, "false_alarm_rate": false_alarms / len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("evals/reports/faults.json"))
    args = parser.parse_args()
    write_json(args.report, {"status": "running", "passed": False})
    try:
        report = run_fault_suite()
    except Exception as error:
        write_json(args.report, {"status": "error", "passed": False,
                                "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    write_json(args.report, report)
    print(f"Synthetic faults detected: {report['detected_count']}/{report['fault_count']}; "
          f"clean-control false alarms: {report['false_alarm_count']}/{report['clean_control_count']}")
    print(f"Report: {args.report}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
