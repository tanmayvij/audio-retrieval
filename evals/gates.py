"""Absolute quality floors and explicit, immutable baseline comparisons."""

import math


TOLERANCE = 1e-12
MINIMUM_RECALL = 0.80
IDENTITY_FIELDS = ("embedding_model", "query_set_sha256", "corpus_sha256")


def make_baseline(report: dict) -> dict:
    """Extract a reviewable baseline; callers must explicitly choose to save it."""
    return {
        "schema_version": 1,
        "source_created_at": report["created_at"],
        **{key: report[key] for key in IDENTITY_FIELDS},
        "modes": {
            mode: {
                "mean_recall": result["mean_recall"],
                "queries": [{"id": row["id"], "recall": row["recall"]} for row in result["queries"]],
            }
            for mode, result in report["modes"].items()
        },
    }


def validate_baseline(baseline: dict, report: dict, query_ids: set[str]) -> None:
    try:
        if baseline["schema_version"] != 1:
            raise ValueError("Unsupported baseline schema version.")
        for key in IDENTITY_FIELDS:
            if baseline[key] != report[key]:
                raise ValueError(f"Baseline {key} mismatch; deliberately replace the baseline after review.")
        for mode in ("semantic", "hybrid"):
            result = baseline["modes"][mode]
            rows = result["queries"]
            if len(rows) != len(query_ids) or {row["id"] for row in rows} != query_ids:
                raise ValueError(f"Baseline {mode} query IDs do not match.")
            for k in ("1", "3", "5"):
                scores = [row["recall"][k] for row in rows]
                values = [*scores, result["mean_recall"][k]]
                if any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError(f"Baseline {mode} contains invalid recall scores.")
                if abs(sum(scores) / len(scores) - result["mean_recall"][k]) > TOLERANCE:
                    raise ValueError(f"Baseline {mode} mean does not match its per-query scores.")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed baseline artifact.") from error


def gate_mode(means: dict, details: list[dict], baseline: dict | None) -> dict:
    reasons = []
    for k in ("3", "5"):
        if means[k] + TOLERANCE < MINIMUM_RECALL:
            reasons.append(f"recall_at_{k}_below_floor")
    zero_hits = [row["id"] for row in details if row["recall"]["3"] == 0]
    comparison = None
    if baseline is not None:
        previous = {row["id"]: row for row in baseline["queries"]}
        delta = means["3"] - baseline["mean_recall"]["3"]
        new_zero_hits = [qid for qid in zero_hits if previous[qid]["recall"]["3"] > 0]
        if delta < -TOLERANCE:
            reasons.append("mean_recall_at_3_regressed")
        if new_zero_hits:
            reasons.append("new_zero_hit_queries_at_3")
        comparison = {
            "baseline_mean_recall_at_3": baseline["mean_recall"]["3"],
            "mean_recall_at_3_delta": delta,
            "new_zero_hit_queries_at_3": new_zero_hits,
            "query_changes": [
                {"id": row["id"], "baseline_recall_at_3": previous[row["id"]]["recall"]["3"],
                 "recall_at_3": row["recall"]["3"],
                 "delta": row["recall"]["3"] - previous[row["id"]]["recall"]["3"]}
                for row in details
            ],
        }
    return {"passed": not reasons, "gate_reasons": reasons,
            "zero_hit_queries_at_3": zero_hits, "baseline_comparison": comparison}
