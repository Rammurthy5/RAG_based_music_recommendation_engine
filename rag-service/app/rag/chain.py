"""LCEL RAG chain for music recommendations.

Pipeline: embed query → nearVector retrieval → format context → prompt → Claude → parse JSON

Graceful degradation:
  - LLM fails / circuit open → return retrieval-only results with source="retrieval_only"
  - Weaviate fails → return cached fallback with source="fallback_cache"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import pybreaker
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.models.schemas import (
    CostInfo,
    RAGConfigInfo,
    Recommendation,
    RecommendResponse,
    ResponseMetadata,
)
from app.rag.prompts import PROMPT_ID, recommendation_prompt
from app.rag.vectorstore import RetrievedTrack, search_tracks
from app.resilience import (
    get_fallback_recommendations,
    llm_breaker,
    weaviate_retry,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-sonnet-4-20250514"

# Cost per token (Claude Sonnet pricing as of 2025)
_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000  # $3 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


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
    recommendations: list[dict], max_per_artist: int = 2
) -> list[dict]:
    """Limit to max_per_artist songs per artist."""
    artist_count: dict[str, int] = {}
    filtered: list[dict] = []
    for rec in recommendations:
        artist = rec.get("artist", "").lower()
        count = artist_count.get(artist, 0)
        if count < max_per_artist:
            filtered.append(rec)
            artist_count[artist] = count + 1
    return filtered


def _parse_llm_output(text: str) -> list[dict]:
    """Parse JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON output: %s", text[:200])
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
            reason=f"Matched your vibe with {t.similarity_score:.0%} similarity.",
            track_id=t.musicbrainz_id,
            similarity_score=t.similarity_score,
        )
        for t in tracks[:limit]
    ]
    return RecommendResponse(
        query=query,
        recommendations=recs,
        metadata=ResponseMetadata(
            source="retrieval_only",
            prompt_id=PROMPT_ID,
            model="",
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
    recs = [
        Recommendation(
            title=r["title"],
            artist=r["artist"],
            album=r.get("album", ""),
            genre=r.get("genre", []),
            reason=r.get("reason", ""),
        )
        for r in fallback[:limit]
    ]
    return RecommendResponse(
        query=query,
        recommendations=recs,
        metadata=ResponseMetadata(
            source="fallback_cache",
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
    except Exception as exc:
        logger.error("Weaviate retrieval failed: %s", exc)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _fallback_response(query, limit, latency_ms)

    if not tracks:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _fallback_response(query, limit, latency_ms)

    # --- Step 2: Invoke LLM via circuit breaker ---
    context = _format_context(tracks)

    try:

        @llm_breaker
        def _call_llm() -> dict:
            llm = ChatAnthropic(
                model=MODEL_NAME,
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout,
                max_tokens=2048,
            )
            chain = recommendation_prompt | llm | StrOutputParser()
            result = chain.invoke(
                {"query": query, "limit": limit, "context": context}
            )
            return {"text": result, "llm": llm}

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
    parsed = _enforce_diversity(parsed)

    recommendations = [
        Recommendation(
            title=r.get("title", ""),
            artist=r.get("artist", ""),
            album=r.get("album", ""),
            genre=r.get("genre", []),
            reason=r.get("reason", ""),
            similarity_score=next(
                (
                    t.similarity_score
                    for t in tracks
                    if t.title.lower() == r.get("title", "").lower()
                ),
                0.0,
            ),
            track_id=next(
                (
                    t.musicbrainz_id
                    for t in tracks
                    if t.title.lower() == r.get("title", "").lower()
                ),
                "",
            ),
        )
        for r in parsed[:limit]
    ]

    # Token usage from the LLM (approximate via last call metadata)
    input_tokens = 0
    output_tokens = 0
    try:
        # langchain-anthropic exposes usage via callback; we estimate here
        input_tokens = llm_instance._get_num_tokens(context + query)
        output_tokens = llm_instance._get_num_tokens(raw_text)
    except Exception:
        pass

    total_cost = (
        input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    return RecommendResponse(
        query=query,
        recommendations=recommendations,
        metadata=ResponseMetadata(
            source="full_rag",
            prompt_id=PROMPT_ID,
            model=MODEL_NAME,
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
