"""Add speaker attribution to transcript chunks."""

from alembic import op
import sqlalchemy as sa


revision = "20260926_02"
down_revision = "20260926_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the required recording-local speaker label."""
    op.add_column(
        "chunks",
        sa.Column("speaker", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    """Remove speaker attribution from transcript chunks."""
    op.drop_column("chunks", "speaker")
