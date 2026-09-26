"""Exploratory score separation, never a production abstention policy."""

import json
import math
from pathlib import Path
from statistics import mean, median


CATEGORIES = {"unrelated", "near_match"}


def load_unanswerable_queries(path: Path, answerable_ids: set[str]) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Unanswerable queries must be a nonempty JSON array.")
    seen = set(answerable_ids)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each unanswerable query must be an object.")
        for field in ("id", "query", "rationale"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Unanswerable query needs a nonblank {field}.")
        if row["id"] in seen:
            raise ValueError("Answerable and unanswerable query IDs must be unique.")
        seen.add(row["id"])
        if row.get("category") not in CATEGORIES:
            raise ValueError("Unanswerable category must be unrelated or near_match.")
        if "relevant_chunk_ids" in row and row["relevant_chunk_ids"] != []:
            raise ValueError("Unanswerable queries cannot have relevant chunk IDs.")
        if row["category"] == "near_match" and (
            not isinstance(row.get("filename"), str) or not row["filename"].strip()
        ):
            raise ValueError("Near-match queries must identify their source filename.")
    if {row["category"] for row in rows} != CATEGORIES:
        raise ValueError("Include both unrelated and near_match unanswerable queries.")
    return rows


def semantic_score(results: list[dict], query_id: str) -> float:
    if not results:
        raise ValueError(f"{query_id}: empty semantic results are a retrieval error, not no-answer detection.")
    scores = [row.get("cosine_similarity") for row in results]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not -1.000001 <= v <= 1.000001 for v in scores):
        raise ValueError(f"{query_id}: invalid semantic cosine similarity.")
    return max(scores)


def _distribution(scores: list[float]) -> dict:
    return {"count": len(scores), "min": min(scores), "median": median(scores),
            "mean": mean(scores), "max": max(scores), "sorted_scores": sorted(scores)}


def analyze_scores(answerable: list[dict], unanswerable: list[dict]) -> dict:
    if not answerable or not unanswerable:
        raise ValueError("Score separation requires both answerable and unanswerable queries.")
    groups = {"all": unanswerable, **{
        category: [row for row in unanswerable if row["category"] == category]
        for category in sorted(CATEGORIES)
    }}
    if any(not rows for rows in groups.values()):
        raise ValueError("Score separation requires both negative categories.")
    for row in [*answerable, *unanswerable]:
        semantic_score([{"cosine_similarity": row["score"]}], row["id"])
    scores = sorted({row["score"] for row in [*answerable, *unanswerable]})
    thresholds = [math.nextafter(scores[0], -math.inf), *scores,
                  math.nextafter(scores[-1], math.inf)]
    sweep = []
    for threshold in thresholds:
        false_alarms = sum(row["score"] < threshold for row in answerable)
        sweep.append({
            "threshold": threshold,
            "false_alarms": false_alarms, "answerable_count": len(answerable),
            "false_alarm_rate": false_alarms / len(answerable),
            "unanswerable": {
                name: {"detected": sum(row["score"] < threshold for row in rows),
                       "count": len(rows),
                       "detection_rate": sum(row["score"] < threshold for row in rows) / len(rows)}
                for name, rows in groups.items()
            },
        })
    return {
        "status": "exploratory", "signal": "highest_semantic_cosine_similarity",
        "flag_rule": "score < threshold", "selected_threshold": None,
        "limitation": "Topical similarity does not establish that a passage answers the question. "
                      "This sweep is exploratory, not held-out detector accuracy or a production cutoff.",
        "answerable": {"queries": answerable, "distribution": _distribution([r["score"] for r in answerable])},
        "unanswerable": {"queries": unanswerable, "distributions": {
            name: _distribution([r["score"] for r in rows]) for name, rows in groups.items()
        }},
        "threshold_sweep": sweep,
    }
