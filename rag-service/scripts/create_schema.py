"""Create the MusicRecommendations collection in Weaviate.

Usage:
    docker-compose exec rag-service python -m scripts.create_schema
"""

import sys

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from app.config import settings

COLLECTION_NAME = "MusicRecommendations"


def create_schema(client: weaviate.WeaviateClient) -> None:
    """Create the MusicRecommendations collection with typed properties."""
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="artist", data_type=DataType.TEXT),
            Property(name="album", data_type=DataType.TEXT),
            Property(name="genres", data_type=DataType.TEXT_ARRAY),
            Property(name="release_year", data_type=DataType.INT),
            Property(name="lyrics_excerpt", data_type=DataType.TEXT),
            Property(name="genius_url", data_type=DataType.TEXT),
            Property(name="musicbrainz_id", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
        ],
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")


def main() -> None:
    client = weaviate.connect_to_custom(
        http_host=settings.weaviate_host,
        http_port=settings.weaviate_http_port,
        http_secure=False,
        grpc_host=settings.weaviate_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=False,
    )
    try:
        if not client.is_ready():
            print("Weaviate is not ready. Aborting.", file=sys.stderr)
            sys.exit(1)
        create_schema(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
