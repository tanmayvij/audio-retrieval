"""Persistence operations for recordings."""

from uuid import UUID, uuid4

from sqlalchemy import insert
from sqlalchemy.engine import Connection

from app.models.recording import Recording


def insert_recording(connection: Connection, filename: str) -> UUID:
    """Insert a recording and return its generated identifier."""
    recording_id = uuid4()
    connection.execute(
        insert(Recording).values(id=recording_id, filename=filename)
    )
    return recording_id
