"""HuggingFace embedding setup.

Uses all-MiniLM-L6-v2 (384 dims, cosine distance) for local embedding.
Shared between ingestion and query time — the same model must be used for both.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a singleton SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return a list of float vectors."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Embed a single text and return its float vector."""
    return embed_texts([text])[0]
