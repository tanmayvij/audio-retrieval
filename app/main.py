"""Application entry point and seed-data readiness check.

Run this module after migrations and data ingestion:

    python3 -m app.main
"""

from sqlalchemy import exists, select

from app.models.chunk import Chunk
from app.services.postgres import connect


def ensure_seeded_data() -> None:
    """Raise when ingestion has not produced searchable data yet."""
    connection = connect()
    try:
        has_seeded_data = connection.scalar(select(exists().where(Chunk.id.is_not(None))))
    finally:
        connection.close()

    if not has_seeded_data:
        raise RuntimeError(
            "Ingestion is pending. Run `python3 app/scripts/init-data.py` first."
        )


def main() -> None:
    """Verify that the application can serve seeded search data."""
    ensure_seeded_data()


if __name__ == "__main__":
    main()
