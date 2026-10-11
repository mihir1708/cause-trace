-- Retrieval passages: one row per (event, section, chunk) with dense and lexical indexes.
-- Passages hold GOLD_ONLY text (analysis, probable cause) of the events they come from, so
-- every search filters on event_date and the target event in SQL (docs/DATA_LEAKAGE.md).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE passages (
    passage_id         text PRIMARY KEY,   -- '<ev_id>:<section_type>:<chunk_index>'
    event_id           text NOT NULL REFERENCES events (ev_id) ON DELETE CASCADE,
    event_date         date NOT NULL,
    section_type       text NOT NULL CHECK (section_type IN ('factual', 'analysis', 'probable_cause')),
    chunk_index        integer NOT NULL CHECK (chunk_index >= 0),
    aircraft_category  text,
    phase_of_flight    text,
    content            text NOT NULL,
    content_sha256     text NOT NULL,
    tsv                tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding          vector(384),
    embedding_model    text,
    embedding_revision text,
    UNIQUE (event_id, section_type, chunk_index)
);

CREATE INDEX passages_embedding_hnsw_idx ON passages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX passages_tsv_gin_idx ON passages USING gin (tsv);
CREATE INDEX passages_event_date_idx ON passages (event_date, event_id);

-- Embeddings keyed by model, revision and content hash, so `make index` can resume and
-- unchanged text is never embedded twice.
CREATE TABLE embedding_cache (
    model          text NOT NULL,
    revision       text NOT NULL,
    content_sha256 text NOT NULL,
    embedding      vector(384) NOT NULL,
    PRIMARY KEY (model, revision, content_sha256)
);

-- Document frequency of each lexeme over historical-split passages, used to pick the
-- salient terms of a lexical query.
CREATE TABLE lexeme_stats (
    lexeme text PRIMARY KEY,
    ndoc   integer NOT NULL
);
