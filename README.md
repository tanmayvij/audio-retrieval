# Recording Search Setup

## Prerequisites

- Python 3.10 or newer
- `pip`
- `ffmpeg` (optional). The [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) Github repository mentions that it uses [`PyAV`](https://github.com/PyAV-Org/PyAV) to perform audio decoding, which bundles ffmpeg.
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed. The migration enables the `vector` extension in the target database, so the database role must be allowed to create extensions.

Create a PostgreSQL database for the application before continuing. For example:

```sql
CREATE DATABASE recording_search;
```

## Set environment variables

The application reads configuration from shell environment variables. Set `DATABASE_URL` to your PostgreSQL database URL:

```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/recording_search"
```

Optional model and processing settings:

```bash
export WHISPER_MODEL="small"
export WHISPER_COMPUTE="int8"
export WHISPER_DEVICE="cpu"
export EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
export CHUNK_TARGET_WORDS="200"
```

## Install Python dependencies

From the repository root, create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run the Alembic migration

With `DATABASE_URL` set, create the database schema and enable `pgvector`:

```bash
alembic upgrade head
```

## Initialize the sample data

Process the bundled audio files, create transcript chunks and embeddings:

```bash
python3 app/scripts/init-data.py
```

## Start the application

Use the master application entry point to start the application:

```bash
python3 -m app.main
```

This command verifies that ingestion has completed and searchable seed data is
available. If ingestion is still pending, run `python3 app/scripts/init-data.py`
before starting the application.
