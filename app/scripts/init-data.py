"""Create embeddings for the sample audio datasets.

Run with:
    python3 app/scripts/init-data.py
"""

from pathlib import Path
import sys

# Running this file directly (``python3 app/scripts/seed.py``) adds only the
# ``app/scripts`` directory to ``sys.path``.  Make the repository root
# available so the absolute ``app.*`` imports below resolve as intended.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunking import chunk_segments
from app.services.embedding import embed_chunks
from app.services.transcription import generate_segments


DATASETS = [
    PROJECT_ROOT / "datasets" / "output_1.mp3",
    PROJECT_ROOT / "datasets" / "output_2.mp3",
    PROJECT_ROOT / "datasets" / "output_3.mp3",
    PROJECT_ROOT / "datasets" / "output_4.mp3",
    PROJECT_ROOT / "datasets" / "output_5.mp3",
    PROJECT_ROOT / "datasets" / "output_6.mp3",
]


def upload_vectors(vectors: list[dict]) -> None:
    """Upload vectors to pgvector."""
    # TODO: Upload ``vectors`` to the vector database.
    _ = vectors


def seed_dataset(dataset_path: Path) -> None:
    """Transcribe, chunk, embed, and upload one audio dataset."""
    filename = dataset_path.name

    print(f"\nLoading dataset: {filename}")
    if not dataset_path.is_file():
        print(f"Skipping {filename}: file not found.")
        return

    print(f"Transcribing {filename}...")
    segments = generate_segments(str(dataset_path))

    print(f"Generating chunks for {filename}...")
    chunks = chunk_segments(segments)
    print(f"Generated {len(chunks)} chunks for {filename}.")

    if not chunks:
        print(f"No transcribed content found in {filename}; skipping embedding and upload.")
        print(f"Completed {filename}.")
        return

    print(f"Generating embeddings for {filename}...")
    vectors = embed_chunks(chunks)

    print(f"Uploading {len(vectors)} vectors for {filename}...")
    upload_vectors(vectors)

    print(f"Completed {filename}.")


def main() -> None:
    """Seed every dataset in ``DATASETS``."""
    print("Initializing seed...")
    print(f"Found {len(DATASETS)} datasets to process.")

    for dataset_path in DATASETS:
        seed_dataset(dataset_path)

    print("\nSeed completed.")


if __name__ == "__main__":
    main()
