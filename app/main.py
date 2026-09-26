"""Application entry point and seed-data readiness check.

Run this module after migrations and data ingestion:

    python3 -m app.main
"""

from sqlalchemy import exists, select
import sys

from app.models.chunk import Chunk
from app.services.postgres import connect
from app.services.retrieval import search


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


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable"
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def show_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return
    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. {result['filename']}")
        print(f"Recording ID: {result['recording_id']}")
        print(f"Speaker: {result['speaker']}")
        print(f"Time: {format_timestamp(result['start_seconds'])} – "
              f"{format_timestamp(result['end_seconds'])}")
        print(result["text"])


def interactive_search() -> None:
    modes = {"1": "hybrid", "2": "exact", "3": "semantic"}
    while True:
        print("\n1. Hybrid\n2. Exact phrase\n3. Semantic\n0. Exit")
        choice = input("Select mode: ").strip()
        if choice == "0":
            return
        if choice not in modes:
            print("Please select 0, 1, 2, or 3.")
            continue
        query = input("Search query: ").strip()
        while not query:
            print("Search query must not be blank.")
            query = input("Search query: ").strip()
        try:
            show_results(search(query, modes[choice]))
        except Exception as error:
            print(f"Search failed: {error}", file=sys.stderr)


def main() -> int:
    """Check seed readiness, then run the interactive search menu."""
    try:
        ensure_seeded_data()
    except Exception as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1
    try:
        interactive_search()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
