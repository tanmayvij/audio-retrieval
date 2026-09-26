"""Create the pgvector recording schema."""

from alembic import op
import sqlalchemy as sa


revision = "20260926_01"
down_revision = None
branch_labels = None
depends_on = None


class Vector(sa.types.UserDefinedType):
    """The pgvector type, whose DDL has no built-in SQLAlchemy equivalent."""

    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_kwargs: object) -> str:
        return f"vector({self.dimensions})"


def upgrade() -> None:
    """Create the pgvector extension and application tables if absent."""
    # PostgreSQL extensions have no Alembic operation abstraction.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "recordings",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.REAL()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_table(
        "transcripts",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("recording_id", sa.UUID(), sa.ForeignKey("recordings.id")),
        sa.Column("full_text", sa.Text(), nullable=False),
        if_not_exists=True,
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("recording_id", sa.UUID(), sa.ForeignKey("recordings.id")),
        sa.Column("start_seconds", sa.REAL()),
        sa.Column("end_seconds", sa.REAL()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768)),
        if_not_exists=True,
    )


def downgrade() -> None:
    """Intentionally preserve application data when migrations are downgraded."""
    # This migration is deliberately non-destructive.
    pass
