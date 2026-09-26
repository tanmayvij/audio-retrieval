"""Persistence operations for recordings."""

from uuid import UUID, uuid4

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from app.models.recording import Recording


def get_recording_filenames(connection: Connection) -> set[str]:
    """Return the filenames already registered as recordings."""
    return set(connection.execute(select(Recording.filename)).scalars())


def insert_recording(
    connection: Connection, filename: str, duration_seconds: float | None
) -> UUID:
    """Insert a recording and return its generated identifier."""
    recording_id = uuid4()
    connection.execute(
        insert(Recording).values(
            id=recording_id,
            filename=filename,
            duration_seconds=duration_seconds,
        )
    )
    return recording_id
