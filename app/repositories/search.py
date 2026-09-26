"""Parameterized PostgreSQL candidate retrieval."""

from sqlalchemy import text
from sqlalchemy.engine import Connection


_FIELDS = """
    c.id AS chunk_id, c.recording_id, r.filename, c.text,
    c.start_seconds, c.end_seconds, c.speaker
"""
_FROM = "FROM chunks c JOIN recordings r ON r.id = c.recording_id"


def lexical_candidates(connection: Connection, query: str, limit: int) -> list[dict]:
    statement = text(f"""
        SELECT {_FIELDS}, ts_rank_cd(c.search_vector, q.query) AS lexical_score
        {_FROM}
        CROSS JOIN (SELECT plainto_tsquery('english', :query) AS query) q
        WHERE c.search_vector @@ q.query
        ORDER BY lexical_score DESC, c.id
        LIMIT :limit
    """)
    return [dict(row) for row in connection.execute(statement, {"query": query, "limit": limit}).mappings()]


def semantic_candidates(connection: Connection, vector: list[float], limit: int) -> list[dict]:
    statement = text(f"""
        SELECT {_FIELDS}, 1 - (c.embedding <=> CAST(:vector AS vector)) AS cosine_similarity
        {_FROM}
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:vector AS vector), c.id
        LIMIT :limit
    """)
    parameters = {"vector": "[" + ",".join(str(value) for value in vector) + "]", "limit": limit}
    return [dict(row) for row in connection.execute(statement, parameters).mappings()]


def exact_candidates(connection: Connection, pattern: str, limit: int) -> list[dict]:
    statement = text(f"""
        SELECT {_FIELDS}
        {_FROM}
        WHERE c.text ~* :pattern
        ORDER BY c.recording_id, c.start_seconds NULLS LAST, c.id
        LIMIT :limit
    """)
    return [dict(row) for row in connection.execute(statement, {"pattern": pattern, "limit": limit}).mappings()]
