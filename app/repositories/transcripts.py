"""Persistence operations for transcripts."""

from uuid import UUID, uuid4

from sqlalchemy import insert
from sqlalchemy.engine import Connection

from app.models.transcript import Transcript


def insert_transcript(
    connection: Connection, recording_id: UUID, full_text: str
) -> UUID:
    """Insert a completed transcript and return its generated identifier."""
    transcript_id = uuid4()
    connection.execute(
        insert(Transcript).values(
            id=transcript_id,
            recording_id=recording_id,
            full_text=full_text,
        )
    )
    return transcript_id
