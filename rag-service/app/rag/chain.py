"""LCEL RAG chain for music recommendations.

Pipeline: embed query → hybrid retrieval → rerank → format context → prompt → Gemini → parse JSON

Graceful degradation:
  - LLM fails / circuit open → return retrieval-only results with source="retrieval_only"
  - Weaviate fails → return cached fallback with source="fallback_cache"
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import pybreaker
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.models.schemas import (
    CostInfo,
    EvalMetrics,
    RAGConfigInfo,
    Recommendation,
    RecommendResponse,
    ResponseMetadata,
)
from app.rag.evaluation import TrackLabel, compute_eval_metrics, load_eval_examples
from app.rag.prompts import PROMPT_ID, recommendation_prompt
from app.rag.query_intent import build_query_intent
from app.rag.provider_factory import build_llm
from app.rag.vectorstore import RetrievedTrack, search_tracks
from app.resilience import (
    get_fallback_recommendations,
    llm_breaker,
    weaviate_retry,
)

logger = logging.getLogger(__name__)

# Cost per token (Gemini 2.5 Flash pricing)
_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000  # $0.15 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000  # $0.60 per 1M output tokens
_EVAL_EXAMPLES_BY_QUERY: dict[str, list[TrackLabel]] | None = None


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _canonical_track_key(title: str, artist: str) -> str:
    return f"{_normalize_query(title)}::{_normalize_query(artist)}"


def _get_reference_tracks_for_query(query: str) -> list[TrackLabel]:
    """Return labeled reference tracks for a known eval query, if any."""
    global _EVAL_EXAMPLES_BY_QUERY
    if _EVAL_EXAMPLES_BY_QUERY is None:
        try:
            _EVAL_EXAMPLES_BY_QUERY = {
                _normalize_query(example.query): example.reference_tracks
                for example in load_eval_examples()
            }
        except Exception as exc:
            logger.debug("Eval set unavailable: %s", exc)
            _EVAL_EXAMPLES_BY_QUERY = {}
    return _EVAL_EXAMPLES_BY_QUERY.get(_normalize_query(query), [])


def _build_eval_metrics(
    query: str,
    retrieved_tracks: list[RetrievedTrack],
    recommendations: list[Recommendation],
) -> EvalMetrics:
    reference_tracks = _get_reference_tracks_for_query(query)
    return compute_eval_metrics(
        query=query,
        retrieved_tracks=retrieved_tracks,
        recommendations=recommendations,
        reference_tracks=reference_tracks or None,
    )


def _format_context(tracks: list[RetrievedTrack]) -> str:
    """Format retrieved tracks into a numbered list for the prompt."""
    lines: list[str] = []
    for i, t in enumerate(tracks, 1):
        parts = [f'{i}. "{t.title}" by {t.artist}']
        if t.album:
            parts.append(f"   Album: {t.album}")
        if t.genres:
            parts.append(f"   Genres: {', '.join(t.genres)}")
        if t.release_year:
            parts.append(f"   Year: {t.release_year}")
        if t.lyrics_excerpt:
            parts.append(f"   Lyrics: {t.lyrics_excerpt[:200]}")
        parts.append(f"   Similarity: {t.similarity_score:.2f}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _enforce_diversity(
    recommendations: list[Recommendation], max_per_artist: int = 2
) -> list[Recommendation]:
    """Limit to max_per_artist songs per artist."""
    artist_count: dict[str, int] = {}
    filtered: list[Recommendation] = []
    for rec in recommendations:
        artist = rec.artist.lower()
        count = artist_count.get(artist, 0)
        if count < max_per_artist:
            filtered.append(rec)
            artist_count[artist] = count + 1
    return filtered


def _vibe_phrase_for_query(query: str) -> str:
    intent = build_query_intent(query)
    vibe_phrase = intent.compact_hint.strip()
    return vibe_phrase or "your vibe"


def _compose_reason(reason: str, query: str) -> str:
    vibe_phrase = _vibe_phrase_for_query(query)
    base_reason = reason.strip()

    if vibe_phrase.lower() == "your vibe":
        if not base_reason:
            return "Matches your vibe."
        if "vibe" in base_reason.lower():
            return base_reason
        return f"{base_reason.rstrip('.')} It matches your vibe."

    if not base_reason:
        return f"Matches your {vibe_phrase} vibe."

    if vibe_phrase.lower() in base_reason.lower():
        return base_reason

    return f"{base_reason.rstrip('.')} It matches your {vibe_phrase} vibe."


def _match_retrieved_track(
    title: str,
    artist: str,
    tracks: list[RetrievedTrack],
) -> RetrievedTrack | None:
    key = _canonical_track_key(title, artist)
    for track in tracks:
        if _canonical_track_key(track.title, track.artist) == key:
            return track

    if title and not artist:
        normalized_title = _normalize_query(title)
        title_matches = [
            track
            for track in tracks
            if _normalize_query(track.title) == normalized_title
        ]
        if len(title_matches) == 1:
            return title_matches[0]

    return None


def _materialize_recommendations(
    parsed: list[dict],
    tracks: list[RetrievedTrack],
    query: str,
    limit: int,
) -> list[Recommendation]:
    """Ground LLM output in retrieved tracks and backfill missing slots."""
    recommendations: list[Recommendation] = []
    selected_keys: set[str] = set()

    for item in parsed:
        track = _match_retrieved_track(
            item.get("title", ""),
            item.get("artist", ""),
            tracks,
        )
        if track is None:
            continue

        key = _canonical_track_key(track.title, track.artist)
        if key in selected_keys:
            continue

        genre_val = item.get("genre", [])
        if isinstance(genre_val, str):
            genre_val = [g.strip() for g in genre_val.split(",") if g.strip()]

        recommendations.append(
            Recommendation(
                title=track.title,
                artist=track.artist,
                album=track.album,
                genre=genre_val or track.genres,
                reason=_compose_reason(item.get("reason", ""), query),
                similarity_score=track.similarity_score,
                track_id=track.musicbrainz_id,
            )
        )
        selected_keys.add(key)
        if len(recommendations) >= limit:
            break

    if len(recommendations) < limit:
        for track in tracks:
            key = _canonical_track_key(track.title, track.artist)
            if key in selected_keys:
                continue
            recommendations.append(
                Recommendation(
                    title=track.title,
                    artist=track.artist,
                    album=track.album,
                    genre=track.genres,
                    reason=_compose_reason("", query),
                    similarity_score=track.similarity_score,
                    track_id=track.musicbrainz_id,
                )
            )
            selected_keys.add(key)
            if len(recommendations) >= limit:
                break

    return _enforce_diversity(recommendations)[:limit]


def _score_fallback_candidate(query: str, item: dict) -> float:
    intent = build_query_intent(query)
    query_tokens = set(re.findall(r"[a-z0-9]+", _normalize_query(query)))
    genre_value = item.get("genre", [])
    genre_text = " ".join(genre_value) if isinstance(genre_value, list) else str(genre_value)
    text = " ".join(
        part
        for part in [
            item.get("title", ""),
            item.get("artist", ""),
            item.get("album", ""),
            genre_text,
            item.get("reason", ""),
        ]
        if part
    ).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    cue_terms = set(intent.cue_terms)

    score = 0.0
    score += min(len(cue_terms & tokens), 8) * 0.08
    score += min(len(query_tokens & tokens), 4) * 0.06
    if query_tokens & set(re.findall(r"[a-z0-9]+", str(item.get("artist", "")).lower())):
        score += 0.08
    if query_tokens & set(re.findall(r"[a-z0-9]+", str(item.get("title", "")).lower())):
        score += 0.08
    return score


def _fallback_reason(query: str, item: dict) -> str:
    base_reason = item.get("reason", "").strip()
    if base_reason:
        return _compose_reason(base_reason, query)
    return _compose_reason("", query)


def _parse_llm_output(text: str) -> list[dict]:
    """Parse JSON array from LLM response.

    Handles: markdown fences, thinking tags, preamble/postamble text.
    Extracts the first JSON array found anywhere in the response.
    """
    original = text
    text = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON array in the text and parse as many
    # complete objects as possible. This salvages truncated responses.
    start = text.find("[")
    if start != -1:
        candidate = text[start:]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        recovered: list[dict] = []
        decoder = json.JSONDecoder()
        idx = 1  # Skip the opening "["
        while idx < len(candidate):
            while idx < len(candidate) and candidate[idx] in " \t\r\n,":
                idx += 1
            if idx >= len(candidate) or candidate[idx] == "]":
                break
            try:
                value, next_idx = decoder.raw_decode(candidate, idx)
            except json.JSONDecodeError:
                break
            if isinstance(value, dict):
                recovered.append(value)
            idx = next_idx

        if recovered:
            logger.warning(
                "Recovered %d recommendations from partial LLM JSON output.",
                len(recovered),
            )
            return recovered

    logger.warning(
        "Failed to parse LLM JSON output. First 200 chars: %s ... Last 200 chars: %s",
        original[:200],
        original[-200:],
    )
    return []


def _retrieval_only_response(
    query: str,
    tracks: list[RetrievedTrack],
    limit: int,
    latency_ms: int,
) -> RecommendResponse:
    """Build a response from retrieval results only (no LLM)."""
    recs = [
        Recommendation(
            title=t.title,
            artist=t.artist,
            album=t.album,
            genre=t.genres,
            reason=_compose_reason(
                f"Matched the retrieved track with {t.similarity_score:.0%} similarity.",
                query,
            ),
            track_id=t.musicbrainz_id,
            similarity_score=t.similarity_score,
        )
        for t in tracks[:limit]
    ]
    eval_metrics = _build_eval_metrics(query, tracks, recs)
    return RecommendResponse(
        query=query,
        recommendations=recs,
        metadata=ResponseMetadata(
            source="retrieval_only",
            prompt_id=PROMPT_ID,
            model="",
            eval_metrics=eval_metrics,
            rag_config=RAGConfigInfo(
                top_k=settings.top_k,
                similarity_threshold=settings.similarity_threshold,
            ),
            latency_ms=latency_ms,
        ),
    )


def _fallback_response(query: str, limit: int, latency_ms: int) -> RecommendResponse:
    """Build a response from the static fallback cache."""
    fallback = get_fallback_recommendations()
    ranked = sorted(
        fallback,
        key=lambda item: _score_fallback_candidate(query, item),
        reverse=True,
    )
    selected = ranked[:limit]
    retrieved_tracks = [
        RetrievedTrack(
            title=item["title"],
            artist=item["artist"],
            album=item.get("album", ""),
            genres=item.get("genre", []),
            lyrics_excerpt=item.get("reason", ""),
            content=" ".join(
                part
                for part in [
                    item.get("title", ""),
                    item.get("artist", ""),
                    item.get("album", ""),
                    " ".join(item.get("genre", []))
                    if isinstance(item.get("genre", []), list)
                    else str(item.get("genre", "")),
                    item.get("reason", ""),
                ]
                if part
            ),
            distance=0.0,
        )
        for item in selected
    ]
    recs = [
        Recommendation(
            title=r["title"],
            artist=r["artist"],
            album=r.get("album", ""),
            genre=r.get("genre", []),
            reason=_fallback_reason(query, r),
        )
        for r in selected
    ]
    eval_metrics = _build_eval_metrics(query, retrieved_tracks, recs)
    return RecommendResponse(
        query=query,
        recommendations=recs,
        metadata=ResponseMetadata(
            source="fallback_cache",
            eval_metrics=eval_metrics,
            latency_ms=latency_ms,
        ),
    )


async def get_recommendations(query: str, limit: int = 5) -> RecommendResponse:
    """Run the full RAG pipeline with graceful degradation.

    1. Retrieve from Weaviate (with retry)
    2. Format context + invoke Claude via circuit breaker
    3. Parse and return structured response
    Falls back to retrieval-only or cached results on failure.
    """
    start = time.perf_counter()

    # --- Step 1: Retrieve from Weaviate ---
    try:
        tracks = weaviate_retry(search_tracks)(query)
        logger.info(
            "Weaviate returned %d tracks above threshold for query: %s",
            len(tracks),
            query[:80],
        )
    except Exception as exc:
        logger.error("Weaviate retrieval failed: %s", exc)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _fallback_response(query, limit, latency_ms)

    if not tracks:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _fallback_response(query, limit, latency_ms)

    # --- Step 2: Invoke LLM via circuit breaker ---
    context = _format_context(tracks)
    intent = build_query_intent(query)

    try:
        llm_bundle = build_llm()

        @llm_breaker
        def _call_llm() -> dict:
            chain = recommendation_prompt | llm_bundle.llm | StrOutputParser()
            result = chain.invoke(
                {
                    "query": query,
                    "limit": limit,
                    "context": context,
                    "intent_hint": intent.compact_hint or intent.summary or "none",
                }
            )
            return {"text": result, "llm": llm_bundle.llm}

        output = _call_llm()
        raw_text = output["text"]
        llm_instance = output["llm"]

    except pybreaker.CircuitBreakerError:
        logger.warning("LLM circuit breaker is OPEN — returning retrieval-only.")
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _retrieval_only_response(query, tracks, limit, latency_ms)
    except Exception as exc:
        logger.error("LLM invocation failed: %s", exc)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _retrieval_only_response(query, tracks, limit, latency_ms)

    # --- Step 3: Parse and build response ---
    parsed = _parse_llm_output(raw_text)
    if not parsed:
        logger.warning("LLM returned no parseable recommendations; using retrieval-only fallback.")
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _retrieval_only_response(query, tracks, limit, latency_ms)

    recommendations = _materialize_recommendations(parsed, tracks, query, limit)

    # Token usage from the LLM (approximate via last call metadata)
    input_tokens = 0
    output_tokens = 0
    try:
        input_tokens = llm_instance.get_num_tokens(context + query)
        output_tokens = llm_instance.get_num_tokens(raw_text)
    except Exception:
        pass

    total_cost = (
        input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    eval_metrics = _build_eval_metrics(query, tracks, recommendations)

    return RecommendResponse(
        query=query,
        recommendations=recommendations,
        metadata=ResponseMetadata(
            source="full_rag",
            provider=llm_bundle.provider,
            prompt_id=PROMPT_ID,
            model=llm_bundle.model,
            eval_metrics=eval_metrics,
            rag_config=RAGConfigInfo(
                top_k=settings.top_k,
                similarity_threshold=settings.similarity_threshold,
            ),
            cost=CostInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost_usd=round(total_cost, 6),
            ),
            latency_ms=latency_ms,
        ),
    )
