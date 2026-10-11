"""Apply the plain SQL migrations in ``causetrace/migrations`` in filename order.

Usage: python -m causetrace.migrate [--database-url URL]
"""

import argparse
import hashlib
from importlib import resources

from causetrace.db import Connection, connect

_LOCK_ID = 7_250_001  # pg_advisory_xact_lock key, serialises concurrent migrators


def migration_files() -> list[tuple[str, str]]:
    """Return ``(version, sql)`` for each ``NNNN_name.sql`` file, sorted by version."""
    root = resources.files("causetrace").joinpath("migrations")
    files = sorted((f for f in root.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
    return [(f.name.removesuffix(".sql"), f.read_text(encoding="utf-8")) for f in files]


def apply_migrations(conn: Connection) -> list[str]:
    """Apply pending migrations in one transaction; return the versions applied.

    An already-applied migration whose file has changed is an error: write a new migration.
    """
    applied_now: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    text PRIMARY KEY,
                checksum   text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        done: dict[str, str] = dict(
            conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
        )
        for version, sql in migration_files():
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            if version in done:
                if done[version] != checksum:
                    raise RuntimeError(f"migration {version} changed after it was applied")
                continue
            conn.execute(sql.encode("utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                (version, checksum),
            )
            applied_now.append(version)
    return applied_now


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    with connect(args.database_url) as conn:
        applied = apply_migrations(conn)
    print(f"applied {len(applied)} migration(s): {', '.join(applied) or 'none'}")


if __name__ == "__main__":
    main()
