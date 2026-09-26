"""Exact, semantic, and hybrid chunk retrieval."""

import re

from app.repositories.search import exact_candidates, lexical_candidates, semantic_candidates
from app.services.embedding import embed_query
from app.services.postgres import connect


def literal_pattern(query: str) -> str:
    # Escape regex syntax explicitly; PostgreSQL does not use Python's regex dialect.
    parts = [re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", part) for part in query.split()]
    return r"(?<![[:alnum:]_])" + "[[:space:]]+".join(parts) + r"(?![[:alnum:]_])"


def rank_candidates(lexical: list[dict], semantic: list[dict], mode: str) -> list[dict]:
    merged: dict = {}
    for branch, candidates in (("lexical", lexical), ("semantic", semantic)):
        for rank, candidate in enumerate(candidates, 1):
            result = merged.setdefault(candidate["chunk_id"], {
                **candidate, "lexical_rank": None, "semantic_rank": None,
                "cosine_similarity": None, "fusion_score": None,
            })
            result[f"{branch}_rank"] = rank
            if branch == "semantic":
                result["cosine_similarity"] = candidate["cosine_similarity"]
            if mode == "hybrid":
                result["fusion_score"] = (result["fusion_score"] or 0.0) + 1 / (60 + rank)
    results = list(merged.values())
    if mode == "hybrid":
        results.sort(key=lambda row: (-row["fusion_score"], str(row["chunk_id"])))
    return results


def search(query: str, mode: str, *, limit: int = 3) -> list[dict]:
    """Return up to ``limit`` chunks with metadata and ranking diagnostics."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Search limit must be a positive integer.")
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be blank.")
    if mode not in {"hybrid", "exact", "semantic"}:
        raise ValueError("Unknown search mode.")
    vector = embed_query(query) if mode != "exact" else None
    with connect() as connection:
        if mode == "exact":
            return [
                {**row, "lexical_rank": None, "semantic_rank": None,
                 "cosine_similarity": None, "fusion_score": None}
                for row in exact_candidates(connection, literal_pattern(query), limit)
            ]
        candidate_limit = max(50, limit)
        lexical = lexical_candidates(connection, query, candidate_limit) if mode == "hybrid" else []
        semantic = semantic_candidates(connection, vector, candidate_limit)
    return rank_candidates(lexical, semantic, mode)[:limit]
