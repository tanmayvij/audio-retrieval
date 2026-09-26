"""Embedded transcript chunk database model."""

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Computed, ForeignKey, Index, REAL, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from app.models.base import Base


class Vector(UserDefinedType[list[float]]):

    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_kwargs: Any) -> str:
        return f"vector({self.dimensions})"

    def bind_processor(self, _dialect: Any):
        """Serialize vectors using pgvector's text input format."""

        def process(value: list[float] | None) -> str | None:
            if value is None:
                return None
            return "[" + ",".join(str(component) for component in value) + "]"

        return process


class Chunk(Base):
    """A timestamped transcript section and its search embedding."""

    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    recording_id: Mapped[UUID] = mapped_column(
        ForeignKey("recordings.id"), nullable=False
    )
    start_seconds: Mapped[float | None] = mapped_column(REAL)
    end_seconds: Mapped[float | None] = mapped_column(REAL)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    speaker: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english'::regconfig, text)", persisted=True),
    )
