"""Create embeddings for the sample audio datasets.

Run with:
    python3 app/scripts/init-data.py
"""

from pathlib import Path
import sys
from uuid import UUID

from sqlalchemy.engine import Connection

# Running this file directly (``python3 app/scripts/init-data.py``) adds only the
# ``app/scripts`` directory to ``sys.path``.  Make the repository root
# available so the absolute ``app.*`` imports below resolve as intended.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.repositories.chunks import insert_chunks
from app.repositories.recordings import insert_recording
from app.repositories.transcripts import insert_transcript

from app.services.chunking import chunk_segments
from app.services.embedding import embed_chunks
from app.services.postgres import connect
from app.services.transcription import generate_segments


DATASETS_DIR = PROJECT_ROOT / "datasets"


def seed_dataset(
    connection: Connection, recording_id: UUID, dataset_path: Path
) -> None:
    """Transcribe, persist, chunk, embed, and persist one audio dataset."""
    filename = dataset_path.name

    print(f"\nLoading dataset: {filename}")
    print(f"Transcribing {filename}...")
    segments = list(generate_segments(str(dataset_path)))

    full_text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    insert_transcript(connection, recording_id, full_text)
    connection.commit()
    print(f"Inserted transcript for {filename}.")

    print(f"Generating chunks for {filename}...")
    chunks = chunk_segments(segments)
    print(f"Generated {len(chunks)} chunks for {filename}.")

    if not chunks:
        print(f"No transcribed content found in {filename}; skipping embedding.")
        print(f"Completed {filename}.")
        return

    print(f"Generating embeddings for {filename}...")
    vectors = embed_chunks(chunks)

    print(f"Inserting {len(vectors)} chunks for {filename}...")
    insert_chunks(connection, recording_id, vectors)
    connection.commit()

    print(f"Completed {filename}.")


def main() -> None:
    """Seed every file in the datasets directory."""
    print("Initializing seed...")
    datasets = sorted(
        dataset_path
        for dataset_path in DATASETS_DIR.iterdir()
        if dataset_path.is_file()
    )

    print(f"Found {len(datasets)} datasets to process.")

    connection = connect()
    try:
        recordings: list[tuple[UUID, Path]] = []
        for dataset_path in datasets:
            recording_id = insert_recording(connection, dataset_path.name)
            connection.commit()
            recordings.append((recording_id, dataset_path))
            print(f"Inserted recording for {dataset_path.name}.")

        for recording_id, dataset_path in recordings:
            seed_dataset(connection, recording_id, dataset_path)
    finally:
        connection.close()

    print("\nSeed completed.")


if __name__ == "__main__":
    main()
