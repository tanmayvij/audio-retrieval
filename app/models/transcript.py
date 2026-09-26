"""Transcript database model."""

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Transcript(Base):
    """The complete transcript generated for a recording."""

    __tablename__ = "transcripts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    recording_id: Mapped[UUID] = mapped_column(ForeignKey("recordings.id"), nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
