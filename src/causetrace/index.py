"""Build the retrieval index for everything in the database.

Steps: rebuild passages from narratives, refresh lexical term statistics, then embed
passages in committed batches (cached by model, revision and text, so an interrupted run
resumes). Needs the ``ml`` dependency group for embeddings.

Usage: python -m causetrace.index [--batch-size N] [--skip-embeddings] [--database-url URL]
"""

import argparse

from causetrace.db import Connection, connect
from causetrace.embeddings import DEFAULT_BATCH_SIZE, Embedder, EmbedStats, embed_passages
from causetrace.migrate import apply_migrations
from causetrace.passages import build_passages
from causetrace.retrieval import refresh_lexeme_stats


def build_index(
    conn: Connection,
    embedder: Embedder | None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
) -> tuple[dict[str, int], int, EmbedStats | None]:
    """Passages per section, lexeme count and embedding stats (None when skipped)."""
    apply_migrations(conn)
    conn.commit()
    sections = build_passages(conn)
    lexemes = refresh_lexeme_stats(conn)
    stats = None
    if embedder is not None:
        stats = embed_passages(conn, embedder, batch_size=batch_size, max_batches=max_batches)
    return sections, lexemes, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build passages, lexical stats and embeddings.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    embedder: Embedder | None = None
    if not args.skip_embeddings:
        from causetrace.embeddings import SentenceTransformerEmbedder

        st = SentenceTransformerEmbedder(batch_size=args.batch_size)
        print(f"embedding with {st.model_name}@{st.revision} on {st.device}")
        embedder = st
    with connect(args.database_url) as conn:
        sections, lexemes, stats = build_index(conn, embedder, batch_size=args.batch_size)
    print("passages: " + ", ".join(f"{k} {v}" for k, v in sections.items()))
    print(f"lexemes: {lexemes}")
    if stats is not None:
        print(
            f"embeddings: {stats.from_cache} from cache, {stats.embedded_texts} texts embedded,"
            f" {stats.remaining} remaining"
        )


if __name__ == "__main__":
    main()
