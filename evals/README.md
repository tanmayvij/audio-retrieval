# Retrieval evaluation

This suite measures chunk-level Recall@1, Recall@3, and Recall@5 for semantic
and hybrid search. Each mode must independently achieve mean Recall@3 and
Recall@5 >= 0.80, maintain baseline mean Recall@3, and introduce no new queries
with zero relevant results in the top three. Comparisons allow only `1e-12`
floating-point tolerance. Partial per-query regressions are reported even when
improvements elsewhere offset them.
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

The ordinary test run includes synthetic fault detection and skips live
evaluations. Explicit runs fail on missing or invalid labels, incompatible
baselines, unavailable databases/models, or retrieval errors. Each answerable
query is retrieved once per mode with `limit=5`; the three prefixes are scored
from that ranking. Negative queries run once in semantic mode. Fewer than k
results are valid; results with no relevant IDs score zero recall. Empty semantic
results are an error for score-separation analysis, never evidence of successful
unanswerable-query detection.

The terminal shows aggregate scores. `evals/reports/recall.json` contains
per-query scores, retrieved and missed IDs at each k, model name, corpus/query
hashes, separate gate outcomes and reasons, baseline deltas, zero-hit query IDs,
and unanswerable score analysis. Reports have `schema_version: 2` and a `status`
of `running`, `success`, `failed` (quality gates), or `error` (setup/data/retrieval).
The runner writes a fresh running status before loading evaluation inputs and
saves results before assertions. Exceptions save a fresh error report before
being re-raised. A killed process may leave `running`; only `success` is a pass.

Override file locations with `--eval-queries PATH`, `--eval-report PATH`,
`--eval-baseline PATH`, and `--eval-unanswerable-queries PATH`. Report output must
not overwrite any input. Low-level `evaluate()` accepts optional keyword-only
`baseline` and `unanswerable` arguments for isolated tests; the live pytest run
always supplies both.
Run the same explicit command in CI only when the fixed database and model are
available, and retain the report as an artifact even on threshold failures.
Scores depend on label completeness. They assess retrieval of stored chunks,
not transcription or diarization accuracy. Exact phrase search is not scored.

## Initial baseline

On the initial 20-query set and 113-chunk corpus, using
`BAAI/bge-base-en-v1.5`, the saved original results were:

| Mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| Semantic | 0.575 | 0.825 | 0.925 |
| Hybrid | 0.575 | 0.825 | 0.925 |

Both modes miss one of the two relevant chunks at k=5 for q14, q18, and q19.
At k=3, q13 has zero relevant hits in both modes. This existing miss remains
visible but does not count as a new regression. These are historical results,
not a claim that the expanded live suite has been run.

`baseline.json` is derived from that saved report, retaining its timestamp,
corpus/query hashes, model name, means, and per-query scores. Evaluation never
updates it. Any change to the corpus IDs/text, answerable query set, or model
name blocks comparison rather than silently accepting a new baseline.

After an intentional corpus/model/label change, review the new labels and
capture a candidate baseline explicitly with the following Python code in the
repository root. It requires the configured database and model:

```python
from pathlib import Path
from app.config import EMBEDDING_MODEL
from app.services.retrieval import search
from evals.gates import make_baseline
from evals.runner import evaluate, format_summary, load_corpus, load_queries, write_json

report = evaluate(load_queries(Path("evals/queries.json")), load_corpus(), search, EMBEDDING_MODEL)
print(format_summary(report))
assert report["passed"], "Candidate must meet absolute Recall@3 and Recall@5 floors"
write_json(Path("evals/reports/baseline-candidate.json"), make_baseline(report))
```

Review every regression before deliberately replacing `evals/baseline.json`
with that candidate and committing it. Then run the full live suite, including
negative-query analysis. Never regenerate a baseline simply to turn a regression
green. Fingerprints still do not verify embedding provenance or vector contents.

## Synthetic fault coverage

```bash
python -m evals.faults
```

This writes `evals/reports/faults.json` (override with `--report PATH`) and exits
nonzero on missed faults or false alarms. The same suite runs in ordinary pytest.
It uses a deterministic synthetic corpus and replaces database/model IO at the
candidate-retrieval boundary while exercising real search, ranking, and gates.
It does not edit the database, execute SQL, or validate the embedding model.

| Injected fault | Expected signal |
| --- | --- |
| Displace relevant semantic candidates | Semantic mean Recall@3 regression |
| Remove one recording's candidates | New hybrid zero-hit query at k=3 |
| Disable lexical branch | New hybrid zero-hit query at k=3 |
| Demote relevant evidence from rank 3 to 4 | Recall@3 regression despite unchanged perfect Recall@5 |
| Remove a labeled chunk's embedding | Explicit missing-embedding validation error |
| Raise a retrieval exception | Query/mode-specific retrieval error with the injected cause |

Each case includes a matched clean run. All six expected signals must occur and
all six clean controls must pass. The report contains counts, detection rate,
false-alarm rate, observed gates/errors, and unexpected failures. Unrelated
crashes never count as successful detection. These rates describe **synthetic
detector coverage**, not production reliability or coverage of all failure modes.

## Unanswerable-query score separation

`unanswerable_queries.json` contains 12 assistant-curated labels based on reading
the full exported catalog: six unrelated topics and six plausible missing facts,
one per recording. Each has an ID, query, category (`unrelated` or `near_match`),
and rationale; near-matches also identify the recording filename. Review these
judgments against the corpus before treating them as human-validated ground
truth, and re-review them whenever the corpus changes.

Negatives do not enter recall denominators. The shared diagnostic is the highest
cosine similarity from semantic retrieval; fusion scores are not probabilities.
The report includes per-query scores/results, distributions for answerable and
negative queries, and a threshold sweep with raw counts and rates:

- Flag weak evidence when `score < threshold` (ties are not flagged).
- Detection rate: fraction of unanswerable queries flagged, overall and by category.
- False-alarm rate: fraction of answerable queries flagged.

The sweep uses every distinct observed score and thresholds immediately below
the minimum and above the maximum, covering flag-none and flag-all operating
points. Empty results, missing scores, non-finite scores, and values outside the
cosine range (allowing `1e-6` numerical error) fail explicitly.

No cutoff is selected, no separation-quality gate is imposed, and interactive
search remains unchanged. High topical similarity can occur even when the
requested fact is absent. These results are exploratory; selecting a detector
later requires a separate calibration set and held-out evaluation, not choosing
and claiming performance on this same sweep.
