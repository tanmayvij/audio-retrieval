from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1) # Load the embedding model once
def _get_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(query: str) -> list[float]:
    """Embed a retrieval query in the same vector space as ingested chunks."""
    if not query.strip():
        raise ValueError("Search query must not be blank.")
    vector = _get_model().encode(
        "Represent this sentence for searching relevant passages: " + query.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    if len(vector) != 768:
        raise ValueError("The embedding model must produce 768-dimensional vectors.")
    return vector


def embed_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return chunks enriched with normalized, JSON-serializable embeddings.

    Each input chunk must include non-empty ``text`` and ``speaker`` values.
    The input chunk dictionaries are copied, so callers can keep their original
    transcript metadata untouched.
    """
    if not chunks:
        return []

    texts = [chunk.get("text", "").strip() for chunk in chunks]
    if any(not text for text in texts):
        raise ValueError("Every chunk must contain non-empty text.")
    if any(not chunk.get("speaker", "").strip() for chunk in chunks):
        raise ValueError("Every chunk must contain a speaker.")

    vectors = _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    return [
        {**chunk, "embedding": vector.tolist()}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
