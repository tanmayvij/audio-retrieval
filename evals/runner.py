"""Label validation, corpus export, evaluation, and report generation."""

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import UUID

from evals.metrics import mean_recall, recall_at_k
from evals.gates import MINIMUM_RECALL, TOLERANCE, gate_mode, validate_baseline
from evals.unanswerable import analyze_scores, load_unanswerable_queries, semantic_score


KS = (1, 3, 5)
MODES = ("semantic", "hybrid")
THRESHOLD = MINIMUM_RECALL


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_queries(path: Path) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError("The query set must be a nonempty JSON array of labeled queries.")
    seen = set()
    for item in queries:
        if not isinstance(item, dict):
            raise ValueError("Each labeled query must be an object.")
        query_id = item.get("id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen:
            raise ValueError("Query IDs must be nonblank, unique strings.")
        seen.add(query_id)
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"{query_id}: query must be a nonblank string.")
        relevant = item.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{query_id}: at least one relevant chunk ID is required.")
        if any(not isinstance(chunk_id, str) for chunk_id in relevant):
            raise ValueError(f"{query_id}: chunk IDs must be UUID strings.")
        try:
            canonical_ids = [str(UUID(chunk_id)) for chunk_id in relevant]
        except ValueError as error:
            raise ValueError(f"{query_id}: chunk IDs must be UUID strings.") from error
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError(f"{query_id}: relevant chunk IDs must be unique.")
        item["relevant_chunk_ids"] = canonical_ids
    return queries


def load_corpus() -> list[dict]:
    # Lazy imports keep metric tests and collection independent of ML packages.
    from sqlalchemy import text

    from app.services.postgres import connect

    with connect() as connection:
        rows = connection.execute(text("""
            SELECT c.id AS chunk_id, c.recording_id, r.filename,
                   c.start_seconds, c.end_seconds, c.speaker, c.text,
                   c.embedding IS NOT NULL AS has_embedding
            FROM chunks c JOIN recordings r ON r.id = c.recording_id
            ORDER BY r.filename, c.start_seconds NULLS LAST, c.id
        """)).mappings()
        return [
            {**dict(row), "chunk_id": str(row["chunk_id"]),
             "recording_id": str(row["recording_id"])}
            for row in rows
        ]


def validate_labels(queries: list[dict], corpus: list[dict]) -> None:
    chunks = {chunk["chunk_id"]: chunk for chunk in corpus}
    for query in queries:
        for chunk_id in query["relevant_chunk_ids"]:
            if chunk_id not in chunks:
                raise ValueError(f"{query['id']}: labeled chunk {chunk_id} is missing; labels may be stale.")
            if not chunks[chunk_id]["has_embedding"]:
                raise ValueError(f"{query['id']}: labeled chunk {chunk_id} has no embedding.")


def evaluate(queries: list[dict], corpus: list[dict], search: Callable, model: str,
             *, baseline: dict | None = None, unanswerable: list[dict] | None = None) -> dict:
    validate_labels(queries, corpus)
    if not queries:
        raise ValueError("Cannot evaluate an empty query set.")
    report = {
        "schema_version": 2,
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "query_set_sha256": fingerprint(queries),
        "corpus_sha256": fingerprint(sorted(
            ({"chunk_id": row["chunk_id"], "text": row["text"]} for row in corpus),
            key=lambda row: row["chunk_id"],
        )),
        "query_count": len(queries),
        "chunk_count": len(corpus),
        "ks": list(KS),
        "gate": {"minimum_mean_recall": {"3": THRESHOLD, "5": THRESHOLD},
                 "baseline_required": baseline is not None, "tolerance": TOLERANCE},
        "modes": {},
    }
    if baseline is not None:
        validate_baseline(baseline, report, {query["id"] for query in queries})
    answerable_scores = []
    for mode in MODES:
        details = []
        for query in queries:
            try:
                results = search(query["query"], mode, limit=max(KS))
            except Exception as error:
                raise RuntimeError(f"Retrieval failed for {mode}, query {query['id']}") from error
            retrieved = [str(row["chunk_id"]) for row in results]
            if mode == "semantic" and unanswerable is not None:
                answerable_scores.append({"id": query["id"], "query": query["query"],
                                          "score": semantic_score(results, query["id"]),
                                          "retrieved_chunk_ids": retrieved})
            relevant = query["relevant_chunk_ids"]
            details.append({
                "id": query["id"], "query": query["query"],
                "relevant_chunk_ids": relevant,
                "retrieved_chunk_ids": retrieved,
                "recall": {str(k): recall_at_k(retrieved, relevant, k) for k in KS},
                "missed_relevant_chunk_ids": {
                    str(k): sorted(set(relevant) - set(retrieved[:k])) for k in KS
                },
            })
        means = {str(k): mean_recall([row["recall"][str(k)] for row in details]) for k in KS}
        report["modes"][mode] = {
            "mean_recall": means, "queries": details,
            **gate_mode(means, details, baseline["modes"][mode] if baseline is not None else None),
        }
    if unanswerable is not None:
        negative_scores = []
        for query in unanswerable:
            try:
                results = search(query["query"], "semantic", limit=max(KS))
            except Exception as error:
                raise RuntimeError(f"Retrieval failed for semantic, query {query['id']}") from error
            negative_scores.append({**query, "score": semantic_score(results, query["id"]),
                                    "retrieved_chunk_ids": [str(row["chunk_id"]) for row in results]})
        report["unanswerable"] = {
            "query_set_sha256": fingerprint(unanswerable),
            **analyze_scores(answerable_scores, negative_scores),
        }
    report["passed"] = all(mode["passed"] for mode in report["modes"].values())
    report["status"] = "success" if report["passed"] else "failed"
    return report


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_evaluation(*, queries_path: Path, baseline_path: Path, unanswerable_path: Path,
                   report_path: Path, corpus_loader: Callable, search: Callable, model: str) -> dict:
    """Persist fresh run status even when setup, data validation, or retrieval fails."""
    if report_path.resolve() in {p.resolve() for p in (queries_path, baseline_path, unanswerable_path)}:
        raise ValueError("Report output must not overwrite evaluation inputs or the baseline.")
    started = datetime.now(timezone.utc).isoformat()
    write_json(report_path, {"schema_version": 2, "created_at": started,
                             "status": "running", "passed": False})
    try:
        queries = load_queries(queries_path)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        negatives = load_unanswerable_queries(unanswerable_path, {q["id"] for q in queries})
        corpus = corpus_loader()
        filenames = {row["filename"] for row in corpus}
        if any(q.get("filename") not in filenames for q in negatives if q["category"] == "near_match"):
            raise ValueError("Near-match source filename is missing from the corpus.")
        report = evaluate(queries, corpus, search, model, baseline=baseline, unanswerable=negatives)
    except Exception as error:
        write_json(report_path, {"schema_version": 2, "created_at": started,
                                 "status": "error", "passed": False,
                                 "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    write_json(report_path, report)
    return report


def format_summary(report: dict) -> str:
    baseline_label = " + baseline" if report["gate"]["baseline_required"] else ""
    lines = [f"Mode        Recall@1  Recall@3  Recall@5  Gates (@3/@5 >= 0.80{baseline_label})"]
    for mode, result in report["modes"].items():
        scores = "  ".join(f"{result['mean_recall'][str(k)]:8.3f}" for k in KS)
        lines.append(f"{mode:10}  {scores}  {'PASS' if result['passed'] else 'FAIL'}")
        lines.append(f"  Zero-hit queries @3: {', '.join(result['zero_hit_queries_at_3']) or 'none'}")
        if result["gate_reasons"]:
            lines.append("  Failed gates: " + ", ".join(result["gate_reasons"]))
        comparison = result["baseline_comparison"]
        if comparison is not None:
            lines.append(f"  Mean Recall@3 change: {comparison['mean_recall_at_3_delta']:+.3f}")
            for row in comparison["query_changes"]:
                if abs(row["delta"]) > TOLERANCE:
                    lines.append(f"  {row['id']}: Recall@3 {row['baseline_recall_at_3']:.3f} -> {row['recall_at_3']:.3f}")
    if "unanswerable" in report:
        analysis = report["unanswerable"]
        lines.append("Unanswerable score separation (exploratory; no selected cutoff or quality gate):")
        distributions = {"answerable": analysis["answerable"]["distribution"],
                         **analysis["unanswerable"]["distributions"]}
        for name, stats in distributions.items():
            lines.append(f"  {name}: n={stats['count']}, cosine min/median/max="
                         f"{stats['min']:.3f}/{stats['median']:.3f}/{stats['max']:.3f}")
        lines.append("  Threshold tradeoffs and per-query scores are in the JSON report.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chunks for manual relevance labeling")
    parser.add_argument("--export-catalog", type=Path, required=True)
    args = parser.parse_args()
    corpus = load_corpus()
    if not corpus:
        raise RuntimeError("No chunks found; complete ingestion before labeling.")
    write_json(args.export_catalog, corpus)
    print(f"Exported {len(corpus)} chunks to {args.export_catalog}")


if __name__ == "__main__":
    main()
