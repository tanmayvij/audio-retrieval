# AI Declaration

I used OpenAI Codex as a coding assistant to implement this project incrementally. I directed the work through specific requirements, asked for explanations and implementation plans, and corrected or refined the approach across successive conversations. Codex generated code, migrations, tests, and documentation, and helped diagnose errors.

My instructions covered the main parts of the project:

- **Audio ingestion:** I specified Whisper's small model with CPU inference, int8 quantization, VAD filtering, and beam size 5. I requested configurable settings, timestamped transcript chunks, embeddings, progress reporting, automatic dataset discovery, and skipping recordings already present in the database.
- **Database structure:** I directed the database setup with schemas for recordings, transcripts, and chunks, and setup of SQLAlchemy, psycopg, and Alembic. I directed Codex to keep schema changes in migrations, database operations in focused repositories, and separation of concerns with connection ownership in the ingestion script.
- **Speakers and retrieval:** I directed a separate diarization service for exactly two speakers, speaker-aware chunks, and persisted speaker labels. I specified an interactive search flow through the main module, with an ingestion pre-check, mode selection, queries, and the top three results including recording ID, timestamps, and speaker.
- **Evaluation:** I requested automated Recall@1, Recall@3, and Recall@5 measurements against a small labeled query set. I clarified that detecting model misbehavior was the main objective, then approved Codex's proposed fault-injection tests, Recall@3 regression checks, and unanswerable-query analysis.

I also used Codex for menial tasks such as reading transcripts from the database and generating sample evaluation queries from their content. I reported runtime issues and requested README documentation with incremental section-wise updates and evaluation commands.