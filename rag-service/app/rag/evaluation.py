"""Lightweight evaluation helpers for recommendation traces.

This module computes the four requested metrics from a single RAG run:
- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision

The recall/precision scores need labeled reference tracks. The text-based
scores use the project's local embedding model so we do not need a separate
evaluation dependency for the first pass.
"""

from __future__ import annotations

import logging
import re
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from app.models.schemas import EvalMetrics, Recommendation
from app.rag.local_embedding import local_embedding
from app.rag.query_intent import build_query_intent
from app.rag.vectorstore import RetrievedTrack
from app.rag.tracing import traced

EmbeddingFn = Callable[[str], list[float]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackLabel:
    """Minimal ground-truth label used for context recall/precision."""

    title: str
    artist: str


@dataclass(frozen=True)
class EvalExample:
    """One labeled evaluation example."""

    query: str
    reference_tracks: list[TrackLabel]


_DEFAULT_EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _track_key(title: str, artist: str) -> str:
    return f"{_normalize(title)}::{_normalize(artist)}"


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    score = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, score))


def _recommendation_text(recommendations: Iterable[Recommendation]) -> str:
    parts: list[str] = []
    for rec in recommendations:
        pieces = [rec.reason, rec.title, rec.artist, rec.album, " ".join(rec.genre)]
        parts.append(" ".join(p for p in pieces if p).strip())
    return "\n".join(parts)


def _context_text(tracks: Iterable[RetrievedTrack]) -> str:
    parts: list[str] = []
    for track in tracks:
        pieces = [
            track.title,
            track.artist,
            track.album,
            ", ".join(track.genres),
            track.lyrics_excerpt[:200],
        ]
        parts.append(" ".join(p for p in pieces if p).strip())
    return "\n".join(parts)


def _track_keys_from_recommendations(recommendations: Iterable[Recommendation]) -> set[str]:
    return {
        _track_key(rec.title, rec.artist)
        for rec in recommendations
        if rec.title and rec.artist
    }


def _track_keys_from_context(tracks: Iterable[RetrievedTrack]) -> set[str]:
    return {
        _track_key(track.title, track.artist)
        for track in tracks
        if track.title and track.artist
    }


@traced(name="compute_eval_metrics", run_type="chain")
def compute_eval_metrics(
    query: str,
    retrieved_tracks: list[RetrievedTrack],
    recommendations: list[Recommendation],
    reference_tracks: list[TrackLabel] | None = None,
    embed_fn: EmbeddingFn | None = None,
) -> EvalMetrics:
    """Compute lightweight eval metrics for one recommendation run.

    Faithfulness is approximated as the share of recommended tracks present in
    the retrieved context. Answer relevancy is the semantic similarity between
    the user query and the final recommendation text. Context recall and
    precision require labeled reference tracks.
    """
    if embed_fn is None:
        if os.environ.get("RAG_EVAL_LOCAL_EMBEDDINGS", "0") == "1":
            embed_fn = local_embedding
        else:
            from app.rag.embeddings import embed_single

            embed_fn = embed_single

    if not recommendations:
        return EvalMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=None,
            context_precision=None,
        )

    intent = build_query_intent(query)
    context_keys = _track_keys_from_context(retrieved_tracks)
    recommendation_keys = _track_keys_from_recommendations(recommendations)
    supported_recommendations = len(recommendation_keys & context_keys)
    faithfulness = supported_recommendations / len(recommendation_keys)

    try:
        query_vec = embed_fn(intent.expanded_query)
        answer_vec = embed_fn(_recommendation_text(recommendations))
    except Exception:
        logger.warning(
            "Falling back to local offline embeddings for eval metrics."
        )
        query_vec = local_embedding(intent.expanded_query)
        answer_vec = local_embedding(_recommendation_text(recommendations))
    answer_relevancy = _cosine_similarity(query_vec, answer_vec)

    context_recall: float | None = None
    context_precision: float | None = None
    if reference_tracks:
        reference_keys = {
            _track_key(label.title, label.artist)
            for label in reference_tracks
            if label.title and label.artist
        }
        if reference_keys:
            relevant_retrieved = len(context_keys & reference_keys)
            context_recall = relevant_retrieved / len(reference_keys)
            context_precision = (
                relevant_retrieved / len(context_keys) if context_keys else 0.0
            )

    return EvalMetrics(
        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(answer_relevancy, 4),
        context_recall=round(context_recall, 4) if context_recall is not None else None,
        context_precision=round(context_precision, 4)
        if context_precision is not None
        else None,
    )


def load_eval_examples(path: str | Path | None = None) -> list[EvalExample]:
    """Load labeled evaluation examples from JSON.

    The file format is:
    [
      {
        "query": "...",
        "reference_tracks": [
          {"title": "...", "artist": "..."}
        ]
      }
    ]
    """
    eval_path = Path(path) if path is not None else _DEFAULT_EVAL_SET_PATH
    raw = json.loads(eval_path.read_text())
    examples: list[EvalExample] = []
    for item in raw:
        references = [
            TrackLabel(title=track["title"], artist=track["artist"])
            for track in item.get("reference_tracks", [])
            if track.get("title") and track.get("artist")
        ]
        examples.append(
            EvalExample(
                query=item["query"],
                reference_tracks=references,
            )
        )
    return examples


def summarize_eval_metrics(metrics: Iterable[EvalMetrics]) -> EvalMetrics:
    """Average a list of per-query metrics into one overall summary."""
    items = list(metrics)
    if not items:
        return EvalMetrics()

    def mean(field: str) -> float | None:
        values = [
            getattr(item, field)
            for item in items
            if getattr(item, field) is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return EvalMetrics(
        faithfulness=mean("faithfulness"),
        answer_relevancy=mean("answer_relevancy"),
        context_recall=mean("context_recall"),
        context_precision=mean("context_precision"),
    )
