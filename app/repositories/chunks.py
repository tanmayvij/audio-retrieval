"""Persistence operations for embedded transcript chunks."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import insert
from sqlalchemy.engine import Connection

from app.models.chunk import Chunk


def insert_chunks(
    connection: Connection,
    recording_id: UUID,
    chunks: Sequence[dict[str, Any]],
) -> None:
    """Insert the completed embedded chunks for one recording."""
    if not chunks:
        return

    connection.execute(
        insert(Chunk),
        [
            {
                "id": uuid4(),
                "recording_id": recording_id,
                "start_seconds": chunk["start"],
                "end_seconds": chunk["end"],
                "text": chunk["text"],
                "speaker": chunk["speaker"],
                "embedding": chunk["embedding"],
            }
            for chunk in chunks
        ],
    )
