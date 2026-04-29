"""Weaviate vector store connection and retrieval.

Provides a singleton Weaviate client and nearVector search against the
MusicRecommendations collection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import weaviate
import weaviate.classes.query as wq

from app.config import settings
from app.rag.embeddings import embed_single

logger = logging.getLogger(__name__)

COLLECTION_NAME = "MusicRecommendations"

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


@dataclass
class RetrievedTrack:
    """A single track retrieved from Weaviate with its metadata."""

    title: str = ""
    artist: str = ""
    album: str = ""
    genres: list[str] = field(default_factory=list)
    release_year: int = 0
    lyrics_excerpt: str = ""
    genius_url: str = ""
    musicbrainz_id: str = ""
    content: str = ""
    distance: float = 1.0

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity (1 - distance)."""
        return max(0.0, 1.0 - self.distance)


def search_tracks(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedTrack]:
    """Embed query and perform nearVector search against Weaviate.

    Returns tracks sorted by similarity, filtered by threshold.
    """
    if top_k is None:
        top_k = settings.top_k
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold

    query_vector = embed_single(query)
    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=wq.MetadataQuery(distance=True),
    )

    tracks: list[RetrievedTrack] = []
    for obj in results.objects:
        props = obj.properties
        distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
        track = RetrievedTrack(
            title=props.get("title", ""),
            artist=props.get("artist", ""),
            album=props.get("album", ""),
            genres=props.get("genres", []),
            release_year=props.get("release_year", 0),
            lyrics_excerpt=props.get("lyrics_excerpt", ""),
            genius_url=props.get("genius_url", ""),
            musicbrainz_id=props.get("musicbrainz_id", ""),
            content=props.get("content", ""),
            distance=distance,
        )
        if track.similarity_score >= similarity_threshold:
            tracks.append(track)

    return tracks
