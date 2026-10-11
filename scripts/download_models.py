"""Download the retrieval models into the Hugging Face cache (needs network).

Tests never download: the slow integration test loads these with local_files_only and
skips when they are not cached. CI runs this first and caches ~/.cache/huggingface.

Usage: uv run --group ml python scripts/download_models.py
"""

from huggingface_hub import model_info
from sentence_transformers import CrossEncoder, SentenceTransformer

from causetrace.config import EMBEDDING_MODEL, EMBEDDING_REVISION, RERANK_MODEL, RERANK_REVISION


def main() -> None:
    SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")
    CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, device="cpu")
    for name, revision in ((EMBEDDING_MODEL, EMBEDDING_REVISION), (RERANK_MODEL, RERANK_REVISION)):
        print(f"{name}@{revision} -> commit {model_info(name, revision=revision).sha}")


if __name__ == "__main__":
    main()
