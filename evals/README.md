# Retrieval evaluation

This suite measures chunk-level Recall@1, Recall@3, and Recall@5 for semantic
and hybrid search. Each mode must independently achieve mean Recall@5 >= 0.80.
Recall is the number of distinct relevant IDs retrieved in the first k results
divided by the total number of labeled relevant IDs. Queries receive equal weight.
For a query with two relevant chunks, Recall@1 cannot exceed 0.50.

## Prerequisites

Install `requirements.txt`, migrate the database, and complete ingestion for
all six recordings using the main README. Set `DATABASE_URL` and use the same
`EMBEDDING_MODEL` used for ingestion (default: `BAAI/bge-base-en-v1.5`). Evaluation
does not run Whisper or diarization; BGE must be cached or downloadable.

Freeze the database while labeling and evaluating. Reingestion creates new chunk
UUIDs and invalidates labels. Keep a database snapshot for reproducibility. The
report records a hash of chunk IDs/text and the model name; this does not verify
which model originally produced stored embeddings or detect changes to vectors.

## Build the labeled query set

`queries.json` contains 20 assistant-curated questions grounded in the 113 chunks
exported from the current six-recording database. Labels were chosen by reading
the complete chunk catalog before running retrieval. Each query includes its
category and a relevance rationale for human review. These are initial judgments,
not independently human-validated ground truth. An empty query set fails explicitly.

Export a readable catalog with chunk IDs, filenames, times, speakers, and text:

```bash
python -m evals.runner --export-catalog evals/catalog.json
```

Read the catalog alongside the full transcripts. Write approximately 20
answerable questions across all six recordings, including factual questions,
paraphrases, and topic queries. For each question, identify all chunks that
independently contain relevant information, including overlapping chunks. Search
results can help discover candidates, but inspect the corpus beyond the current
retriever's results. Human-review the relevance judgments before using the gate.

Put the judgments in `evals/queries.json`:

```json
[
  {
    "id": "q01",
    "query": "A question grounded in an actual transcript",
    "relevant_chunk_ids": ["an-actual-chunk-uuid-from-the-catalog"]
  }
]
```

The example UUID is a placeholder, not a usable label. Query IDs must be unique,
queries nonblank, and relevant IDs valid, unique UUIDs that exist with embeddings.
Keep at least one relevant chunk per query. Unanswerable questions need a separate
metric. More than five relevant chunks puts a ceiling below 1.0 on Recall@5;
never omit valid labels just to improve the score.

## Run and inspect

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m pytest evals/ --run-evals -m eval -s
```

The ordinary test run skips live evaluations. Explicit runs fail on missing or
invalid labels, unavailable databases/models, or retrieval errors. Each query is
retrieved once per mode with `limit=5`; the three prefixes are scored from that
ranking. Fewer than k results are valid, and zero matches score zero.

The terminal shows aggregate scores. `evals/reports/recall.json` contains
per-query scores, retrieved and missed IDs at each k, model name, corpus/query
hashes, and separate gate outcomes. The report is written before threshold
assertions, including when either mode fails. Setup or retrieval errors abort
evaluation without a new score report; an existing report may be from an older
run, so check its timestamp and the pytest outcome.

Override file locations with `--eval-queries PATH` and `--eval-report PATH`.
Run the same explicit command in CI only when the fixed database and model are
available, and retain the report as an artifact even on threshold failures.
Scores depend on label completeness. They assess retrieval of stored chunks,
not transcription or diarization accuracy. Exact phrase search is not scored.

## Initial baseline

On the initial 20-query set and 113-chunk corpus, using
`BAAI/bge-base-en-v1.5`, both modes pass the 0.80 Recall@5 gate:

| Mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| Semantic | 0.575 | 0.825 | 0.925 |
| Hybrid | 0.575 | 0.825 | 0.925 |

Both modes miss one of the two relevant chunks at k=5 for q14, q18, and q19.