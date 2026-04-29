"""Resilience utilities: circuit breaker, retries, and fallback logic.

- Circuit breaker (pybreaker) wraps LLM calls — opens after 5 failures, resets after 60s.
- Retries with jitter (tenacity) for idempotent calls only (Weaviate retrieval, embeddings).
- Graceful degradation: LLM fails → retrieval-only results; Weaviate fails → cached fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pybreaker
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import settings

logger = logging.getLogger(__name__)

# --- Circuit Breaker for LLM calls ---
llm_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_fail_max,
    reset_timeout=settings.cb_reset_timeout,
    name="llm_circuit_breaker",
)

# --- Retry decorator for idempotent operations ---
weaviate_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=5, jitter=1),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

embedding_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.3, max=3, jitter=0.5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

# --- Fallback cache ---
_FALLBACK_PATH = Path(__file__).parent.parent / "data" / "fallback_playlists.json"
_fallback_cache: list[dict] | None = None


def get_fallback_recommendations() -> list[dict]:
    """Return a static set of curated recommendations when all else fails."""
    global _fallback_cache
    if _fallback_cache is not None:
        return _fallback_cache

    if _FALLBACK_PATH.exists():
        try:
            _fallback_cache = json.loads(_FALLBACK_PATH.read_text())
            return _fallback_cache
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load fallback cache: %s", exc)

    # Hardcoded last-resort fallback
    _fallback_cache = [
        {
            "title": "Here Comes the Sun",
            "artist": "The Beatles",
            "album": "Abbey Road",
            "genre": ["rock", "pop"],
            "reason": "A timeless feel-good classic that lifts any mood.",
        },
        {
            "title": "Weightless",
            "artist": "Marconi Union",
            "album": "Weightless",
            "genre": ["ambient"],
            "reason": "Scientifically designed to reduce anxiety and calm the mind.",
        },
        {
            "title": "Happy",
            "artist": "Pharrell Williams",
            "album": "G I R L",
            "genre": ["pop", "soul"],
            "reason": "Pure upbeat energy — impossible not to smile.",
        },
        {
            "title": "Clair de Lune",
            "artist": "Claude Debussy",
            "album": "Suite bergamasque",
            "genre": ["classical"],
            "reason": "Ethereal piano that captures dreamy, reflective moods.",
        },
        {
            "title": "Redbone",
            "artist": "Childish Gambino",
            "album": "Awaken, My Love!",
            "genre": ["funk", "r&b"],
            "reason": "Groovy, psychedelic vibes with a nostalgic warmth.",
        },
    ]
    return _fallback_cache
