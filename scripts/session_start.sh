#!/usr/bin/env bash
# SessionStart hook for Claude Code cloud sessions (no Docker there).
# Installs PostgreSQL 16 + pgvector via apt when missing, starts the cluster,
# creates the causetrace role/databases and syncs Python deps, so `make check` works.
# Idempotent. Does nothing outside cloud sessions (use `make up` locally).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PG_VERSION=16
DB_USER=causetrace
DB_PASSWORD=causetrace
SUDO=""
AS_POSTGRES=(runuser -u postgres --)
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; AS_POSTGRES=(sudo -u postgres); fi

missing=()
[ -x "/usr/lib/postgresql/${PG_VERSION}/bin/postgres" ] || missing+=("postgresql-${PG_VERSION}")
[ -f "/usr/share/postgresql/${PG_VERSION}/extension/vector.control" ] || missing+=("postgresql-${PG_VERSION}-pgvector")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "session_start: installing ${missing[*]}" >&2
  $SUDO apt-get update -qq >&2
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}" >&2
fi

if ! pg_lsclusters -h | awk -v v="$PG_VERSION" '$1 == v && $2 == "main"' | grep -q .; then
  $SUDO pg_createcluster "$PG_VERSION" main >&2
fi
if ! pg_lsclusters -h | awk -v v="$PG_VERSION" '$1 == v && $2 == "main" && $4 == "online"' | grep -q .; then
  $SUDO pg_ctlcluster "$PG_VERSION" main start >&2
fi

psql_admin() { "${AS_POSTGRES[@]}" psql -v ON_ERROR_STOP=1 -qtA "$@"; }
if [ "$(psql_admin -c "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'")" != "1" ]; then
  psql_admin -c "CREATE ROLE ${DB_USER} LOGIN CREATEDB PASSWORD '${DB_PASSWORD}'" >&2
fi
for db in causetrace causetrace_test; do
  if [ "$(psql_admin -c "SELECT 1 FROM pg_database WHERE datname = '${db}'")" != "1" ]; then
    psql_admin -c "CREATE DATABASE ${db} OWNER ${DB_USER}" >&2
  fi
  # pgvector is not a trusted extension, so the superuser enables it up front.
  psql_admin -d "$db" -c "CREATE EXTENSION IF NOT EXISTS vector" >&2
done

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"
if command -v uv >/dev/null 2>&1; then
  uv sync --frozen >&2 || uv sync >&2
fi
echo "session_start: PostgreSQL ${PG_VERSION} with pgvector is up on localhost:5432" >&2
