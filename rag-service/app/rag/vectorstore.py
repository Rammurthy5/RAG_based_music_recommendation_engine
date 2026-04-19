"""Weaviate vector store connection.

Provides a singleton Weaviate client and WeaviateVectorStore for retrieval.
Full implementation in Phase 3 (step 9).
"""

import weaviate

from app.config import settings

_client: weaviate.WeaviateClient | None = None


def get_client() -> weaviate.WeaviateClient:
    """Return a shared Weaviate client, connecting on first call."""
    global _client
    if _client is None:
        _client = weaviate.connect_to_custom(
            http_host=settings.weaviate_host,
            http_port=settings.weaviate_http_port,
            http_secure=False,
            grpc_host=settings.weaviate_host,
            grpc_port=settings.weaviate_grpc_port,
            grpc_secure=False,
        )
    return _client
