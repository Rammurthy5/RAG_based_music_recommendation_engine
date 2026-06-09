"""Weaviate vector store connection and retrieval.

Provides a singleton Weaviate client and nearVector search against the
MusicRecommendations collection with timeout enforcement and embedding retry.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import weaviate
import weaviate.classes.query as wq

from app.config import settings
from app.rag.embeddings import embed_single
from app.rag.local_embedding import local_embedding
from app.rag.query_intent import QueryIntent, build_query_intent
from app.resilience import embedding_retry

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
            additional_config=weaviate.classes.init.AdditionalConfig(
                timeout=weaviate.classes.init.Timeout(
                    query=settings.weaviate_query_timeout,
                    insert=30,
                    init=10,
                ),
            ),
        )
    return _client


def close_client() -> None:
    """Close the shared Weaviate client if open."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


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


def _track_key(track: RetrievedTrack) -> str:
    title = re.sub(r"\s+", " ", track.title.lower()).strip()
    artist = re.sub(r"\s+", " ", track.artist.lower()).strip()
    if track.musicbrainz_id:
        return track.musicbrainz_id
    return f"{title}::{artist}"


def _text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _track_text(track: RetrievedTrack) -> str:
    return " ".join(
        part
        for part in [
            track.title,
            track.artist,
            track.album,
            " ".join(track.genres),
            track.lyrics_excerpt,
            track.content,
        ]
        if part
    ).strip()


def _candidate_score(track: RetrievedTrack, intent: QueryIntent) -> float:
    text_tokens = _text_tokens(_track_text(track))
    query_tokens = _text_tokens(intent.normalized)
    cue_tokens = set(intent.cue_terms)

    score = track.similarity_score
    score += min(len(cue_tokens & text_tokens), 8) * 0.12
    score += min(len(query_tokens & text_tokens), 4) * 0.08

    if query_tokens & _text_tokens(track.artist):
        score += 0.12
    if query_tokens & _text_tokens(track.title):
        score += 0.08
    if query_tokens & _text_tokens(" ".join(track.genres)):
        score += 0.08

    return score


def _merge_candidates(candidates: list[RetrievedTrack]) -> list[RetrievedTrack]:
    merged: dict[str, RetrievedTrack] = {}
    for track in candidates:
        key = _track_key(track)
        existing = merged.get(key)
        if existing is None or track.similarity_score > existing.similarity_score:
            merged[key] = track
    return list(merged.values())


def _rerank_tracks(query: str, tracks: list[RetrievedTrack]) -> list[RetrievedTrack]:
    intent = build_query_intent(query)
    return sorted(tracks, key=lambda track: _candidate_score(track, intent), reverse=True)


def search_tracks(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedTrack]:
    """Embed query and perform nearVector search against Weaviate.

    Returns tracks sorted by similarity, filtered by threshold.
    Embedding is retried on transient failures (cold start, etc.).
    """
    if top_k is None:
        top_k = settings.top_k
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold

    intent = build_query_intent(query)
    candidate_limit = max(top_k * 2, top_k + 4)
    query_variants = [query.strip()]
    if intent.expanded_query.strip() and intent.expanded_query.strip() != query.strip():
        query_variants.append(intent.expanded_query.strip())

    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    raw_candidates: list[RetrievedTrack] = []
    for variant in query_variants:
        if os.environ.get("RAG_EVAL_LOCAL_EMBEDDINGS", "0") == "1":
            query_vector = local_embedding(variant)
        else:
            query_vector = embedding_retry(embed_single)(variant)

        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=candidate_limit,
            return_metadata=wq.MetadataQuery(distance=True),
        )

        for obj in results.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            raw_candidates.append(
                RetrievedTrack(
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
            )

    merged_candidates = _merge_candidates(raw_candidates)
    ranked_candidates = _rerank_tracks(query, merged_candidates)

    tracks = [
        track
        for track in ranked_candidates
        if track.similarity_score >= similarity_threshold
    ]

    if not tracks and ranked_candidates:
        logger.warning(
            "No tracks met threshold %.2f for query '%s'; returning top reranked candidates.",
            similarity_threshold,
            query[:80],
        )
        tracks = ranked_candidates

    return tracks[:top_k]
