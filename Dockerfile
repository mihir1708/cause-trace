# CauseTrace API: FastAPI + the retrieval models (ml group) + the agent.
# Build: docker compose build api. Models are downloaded into the hf-cache volume on first
# use (or mount an existing Hugging Face cache); the database comes from DATABASE_URL.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so source changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group ml --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --group ml

RUN useradd --create-home --uid 1000 causetrace && mkdir -p /models \
    && chown causetrace /models
USER causetrace

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"
CMD ["uvicorn", "causetrace.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
