"""Weaviate vector store connection and retrieval.

Provides a singleton Weaviate client, hybrid search, and cross-encoder reranking
against the MusicRecommendations collection with timeout enforcement and
embedding retry.
"""

from __future__ import annotations

import logging
import math
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
HYBRID_QUERY_PROPERTIES = [
    "title^3",
    "artist^3",
    "album",
    "genres",
    "lyrics_excerpt^2",
    "content^2",
]

_client: weaviate.WeaviateClient | None = None
_cross_encoder: object | None = None


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
    retrieval_score: float = 0.0
    retrieval_mode: str = ""

    @property
    def similarity_score(self) -> float:
        """Return the best available retrieval score as a 0-1-ish similarity."""
        if self.retrieval_score > 0:
            return self.retrieval_score
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

    score = track.retrieval_score if track.retrieval_score > 0 else track.similarity_score
    score += min(len(cue_tokens & text_tokens), 8) * 0.12
    score += min(len(query_tokens & text_tokens), 4) * 0.08

    if query_tokens & _text_tokens(track.artist):
        score += 0.12
    if query_tokens & _text_tokens(track.title):
        score += 0.08
    if query_tokens & _text_tokens(" ".join(track.genres)):
        score += 0.08

    return score


def _build_rerank_query(query: str) -> str:
    intent = build_query_intent(query)
    base_query = intent.expanded_query.strip() or query.strip()
    hint = intent.compact_hint.strip()

    if hint and hint.lower() != "your vibe" and hint.lower() not in base_query.lower():
        return f"{base_query} | vibe: {hint}"

    if intent.summary and intent.summary.lower() not in base_query.lower():
        return f"{base_query} | cues: {intent.summary}"

    return base_query


def _candidate_pool_limit(top_k: int) -> int:
    return max(
        top_k * settings.retrieval_candidate_multiplier,
        settings.retrieval_min_candidate_pool,
    )


def _get_cross_encoder() -> object:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _track_from_object(
    obj: object,
    *,
    retrieval_mode: str,
    retrieval_score: float,
) -> RetrievedTrack:
    props = getattr(obj, "properties", {}) or {}
    metadata = getattr(obj, "metadata", None)
    distance = getattr(metadata, "distance", None)
    if distance is None:
        distance = 1.0

    return RetrievedTrack(
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
        retrieval_score=retrieval_score,
        retrieval_mode=retrieval_mode,
    )


def _normalize_variant_scores(tracks: list[RetrievedTrack]) -> None:
    if not tracks:
        return

    max_score = max(track.retrieval_score for track in tracks)
    if max_score <= 0:
        return

    for track in tracks:
        track.retrieval_score = track.retrieval_score / max_score


def _score_tracks_with_cross_encoder(
    query: str, tracks: list[RetrievedTrack]
) -> list[float]:
    rerank_query = _build_rerank_query(query)
    candidate_pairs = [(rerank_query, _track_text(track)) for track in tracks]
    reranker = _get_cross_encoder()
    raw_scores = reranker.predict(
        candidate_pairs,
        batch_size=settings.cross_encoder_batch_size,
        show_progress_bar=False,
    )

    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    elif not isinstance(raw_scores, list):
        raw_scores = [raw_scores]

    scores = [_sigmoid(float(score)) for score in raw_scores]
    if len(scores) != len(tracks):
        raise ValueError(
            f"Cross-encoder returned {len(scores)} scores for {len(tracks)} tracks"
        )
    return scores


def _query_variant_candidates(
    collection: object,
    variant: str,
    candidate_limit: int,
    retrieval_mode: str,
) -> list[RetrievedTrack]:
    if retrieval_mode == "hybrid":
        results = collection.query.hybrid(
            query=variant,
            vector=embedding_retry(embed_single)(variant)
            if os.environ.get("RAG_EVAL_LOCAL_EMBEDDINGS", "0") != "1"
            else local_embedding(variant),
            alpha=settings.hybrid_alpha,
            query_properties=HYBRID_QUERY_PROPERTIES,
            limit=candidate_limit,
            return_metadata=wq.MetadataQuery(score=True),
        )
        tracks = []
        for obj in results.objects:
            metadata = getattr(obj, "metadata", None)
            score = getattr(metadata, "score", None)
            tracks.append(
                _track_from_object(
                    obj,
                    retrieval_mode="hybrid",
                    retrieval_score=score if score is not None else 0.0,
                )
            )
        _normalize_variant_scores(tracks)
        return tracks

    if os.environ.get("RAG_EVAL_LOCAL_EMBEDDINGS", "0") == "1":
        query_vector = local_embedding(variant)
    else:
        query_vector = embedding_retry(embed_single)(variant)

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=candidate_limit,
        return_metadata=wq.MetadataQuery(distance=True),
    )
    tracks = []
    for obj in results.objects:
        metadata = getattr(obj, "metadata", None)
        distance = getattr(metadata, "distance", None)
        if distance is None:
            distance = 1.0
        tracks.append(
            _track_from_object(
                obj,
                retrieval_mode="vector",
                retrieval_score=max(0.0, 1.0 - distance),
            )
        )
    return tracks


def _merge_candidates(candidates: list[RetrievedTrack]) -> list[RetrievedTrack]:
    merged: dict[str, RetrievedTrack] = {}
    for track in candidates:
        key = _track_key(track)
        existing = merged.get(key)
        if existing is None:
            merged[key] = track
            continue

        existing_score = (
            existing.retrieval_score if existing.retrieval_score > 0 else existing.similarity_score
        )
        track_score = track.retrieval_score if track.retrieval_score > 0 else track.similarity_score
        if track_score > existing_score or (
            track_score == existing_score
            and track.similarity_score > existing.similarity_score
        ):
            merged[key] = track
    return list(merged.values())


def _heuristic_rerank_tracks(query: str, tracks: list[RetrievedTrack]) -> list[RetrievedTrack]:
    intent = build_query_intent(query)
    return sorted(tracks, key=lambda track: _candidate_score(track, intent), reverse=True)


def _rerank_tracks(query: str, tracks: list[RetrievedTrack]) -> list[RetrievedTrack]:
    if not tracks:
        return []

    selected_tracks = tracks
    max_candidates = settings.cross_encoder_max_candidates
    if max_candidates > 0 and len(tracks) > max_candidates:
        selected_tracks = sorted(
            tracks,
            key=lambda track: track.retrieval_score
            if track.retrieval_score > 0
            else track.similarity_score,
            reverse=True,
        )[:max_candidates]
        logger.info(
            "Cross-encoder candidate cap applied: %d -> %d",
            len(tracks),
            len(selected_tracks),
        )

    try:
        scores = _score_tracks_with_cross_encoder(query, selected_tracks)
        for track, score in zip(selected_tracks, scores):
            track.retrieval_score = score
        return sorted(selected_tracks, key=lambda track: track.retrieval_score, reverse=True)
    except Exception as exc:
        logger.warning(
            "Cross-encoder rerank failed for query '%s' (%s); falling back to heuristic rerank.",
            query[:80],
            exc,
        )
        return _heuristic_rerank_tracks(query, tracks)


def search_tracks(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedTrack]:
    """Embed query and perform hybrid search against Weaviate.

    Returns tracks sorted by similarity, filtered by threshold.
    Embedding is retried on transient failures (cold start, etc.).
    """
    if top_k is None:
        top_k = settings.top_k
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold

    intent = build_query_intent(query)
    candidate_limit = _candidate_pool_limit(top_k)
    query_variants = [query.strip()]
    if intent.expanded_query.strip() and intent.expanded_query.strip() != query.strip():
        query_variants.append(intent.expanded_query.strip())

    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    retrieval_mode = "hybrid"
    raw_candidates: list[RetrievedTrack] = []
    try:
        for variant in query_variants:
            variant_candidates = _query_variant_candidates(
                collection,
                variant,
                candidate_limit,
                retrieval_mode="hybrid",
            )
            logger.info(
                "Hybrid retrieval returned %d candidates for variant '%s'",
                len(variant_candidates),
                variant[:80],
            )
            raw_candidates.extend(variant_candidates)
    except Exception as exc:
        logger.warning(
            "Hybrid retrieval failed for query '%s' (%s); falling back to vector-only search.",
            query[:80],
            exc,
        )
        retrieval_mode = "vector_only"
        raw_candidates = []
        for variant in query_variants:
            variant_candidates = _query_variant_candidates(
                collection,
                variant,
                candidate_limit,
                retrieval_mode="vector",
            )
            logger.info(
                "Vector-only retrieval returned %d candidates for variant '%s'",
                len(variant_candidates),
                variant[:80],
            )
            raw_candidates.extend(variant_candidates)

    merged_candidates = _merge_candidates(raw_candidates)
    logger.info(
        "Retrieval mode=%s, variants=%d, raw_candidates=%d, merged_candidates=%d, candidate_limit=%d",
        retrieval_mode,
        len(query_variants),
        len(raw_candidates),
        len(merged_candidates),
        candidate_limit,
    )
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
