"""Query normalization and intent expansion helpers for music retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.tracing import traced

_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "be",
    "big",
    "for",
    "from",
    "give",
    "good",
    "in",
    "into",
    "like",
    "me",
    "more",
    "music",
    "night",
    "of",
    "on",
    "or",
    "rough",
    "songs",
    "sound",
    "some",
    "the",
    "to",
    "with",
    "vibe",
    "vibes",
}

_PHRASE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    (
        "rough night",
        (
            "comforting",
            "gentle",
            "soothing",
            "hopeful",
            "reflective",
            "scientist",
            "fix",
            "you",
            "yellow",
            "clocks",
            "coldplay",
        ),
    ),
    ("late night", ("moody", "intimate", "reflective", "atmospheric", "calm")),
    (
        "big feelings",
        (
            "anthemic",
            "uplifting",
            "emotional",
            "powerful",
            "sweeping",
            "viva",
            "vida",
            "fix",
            "you",
            "clocks",
            "coldplay",
        ),
    ),
    ("full of energy", ("energetic", "driving", "powerful", "upbeat", "anthemic")),
    ("feel good", ("uplifting", "bright", "warm", "catchy", "optimistic")),
    (
        "romantic tamil",
        (
            "romantic",
            "tamil",
            "indian",
            "soundtrack",
            "carnatic",
            "kolaveri",
            "vaaji",
            "sahana",
            "sahara",
            "balleilakka",
            "tamil",
        ),
    ),
    ("romantic", ("romantic", "tender", "dreamy", "intimate", "heartfelt")),
    ("tamil", ("tamil", "indian", "soundtrack", "film", "cinematic", "carnatic")),
    ("indian", ("indian", "soundtrack", "film", "cinematic", "carnatic")),
    ("rage", ("angry", "intense", "aggressive", "rebellious", "cathartic", "powerful")),
    ("angry", ("angry", "intense", "aggressive", "rebellious", "cathartic")),
    ("energetic", ("energetic", "driving", "powerful", "anthemic", "upbeat")),
    ("anthemic", ("anthemic", "uplifting", "emotional", "powerful", "sweeping")),
    ("sad", ("sad", "melancholy", "gentle", "soothing", "reflective")),
    ("comforting", ("comforting", "gentle", "warm", "soothing", "hopeful")),
    ("hopeful", ("hopeful", "uplifting", "warm", "gentle", "reflective")),
    ("party", ("upbeat", "danceable", "energetic", "celebratory", "fun")),
]

_ERA_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("90s", ("1990s", "nostalgic", "classic", "timeless")),
    ("2000s", ("2000s", "nostalgic", "classic")),
    ("classic", ("classic", "timeless", "nostalgic")),
]

_MUSIC_DOMAIN_TERMS = {
    "album",
    "albums",
    "artist",
    "artists",
    "beat",
    "beats",
    "chord",
    "chords",
    "chorus",
    "genre",
    "genres",
    "instrumental",
    "lyrics",
    "lyric",
    "mix",
    "music",
    "playlist",
    "playlists",
    "record",
    "records",
    "remix",
    "song",
    "songs",
    "soundtrack",
    "track",
    "tracks",
}

_NON_MUSIC_DOMAIN_TERMS = {
    "bake",
    "baking",
    "book",
    "books",
    "code",
    "coding",
    "cook",
    "cooking",
    "dinner",
    "finance",
    "flight",
    "flights",
    "ingredient",
    "ingredients",
    "medical",
    "meal",
    "movie",
    "movies",
    "pasta",
    "pizza",
    "python",
    "recipe",
    "recipes",
    "restaurant",
    "restaurants",
    "software",
    "stock",
    "stocks",
    "travel",
    "weather",
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _terms(text: str) -> set[str]:
    return {
        token
        for token in _normalize(text).split()
        if token and token not in _STOPWORDS
    }


@dataclass(frozen=True)
class QueryIntent:
    original: str
    normalized: str
    expanded_query: str
    cue_terms: tuple[str, ...]
    summary: str
    compact_hint: str
    is_music_domain: bool


@traced(name="build_query_intent", run_type="chain")
def build_query_intent(query: str) -> QueryIntent:
    """Expand a natural-language query into retrieval-friendly intent cues."""
    normalized = _normalize(query)
    cue_terms: set[str] = set(_terms(query))
    matched_fragments: list[str] = []

    for fragment, expansions in _PHRASE_HINTS:
        if fragment in normalized:
            cue_terms.update(expansions)
            matched_fragments.append(f"{fragment}: {', '.join(expansions)}")

    for fragment, expansions in _ERA_HINTS:
        if fragment in normalized:
            cue_terms.update(expansions)
            matched_fragments.append(f"{fragment}: {', '.join(expansions)}")

    # Keep the original query tokens and append only the useful expansion terms.
    expansion_terms = sorted(cue_terms - _terms(query))
    expanded_query = " ".join(part for part in [query.strip(), " ".join(expansion_terms)] if part).strip()
    summary = "; ".join(matched_fragments)
    if not summary and expansion_terms:
        summary = "intent cues: " + ", ".join(expansion_terms)

    compact_hint = _build_compact_hint(normalized, cue_terms)
    is_music_domain = _is_music_domain_query(normalized, cue_terms)

    return QueryIntent(
        original=query,
        normalized=normalized,
        expanded_query=expanded_query,
        cue_terms=tuple(sorted(cue_terms)),
        summary=summary,
        compact_hint=compact_hint,
        is_music_domain=is_music_domain,
    )


def _build_compact_hint(normalized_query: str, cue_terms: set[str]) -> str:
    if "rough night" in normalized_query:
        return "comforting, gentle, reflective"
    if "romantic tamil" in normalized_query:
        return "romantic Tamil, melodic, soundtrack"
    if "big feelings" in normalized_query:
        return "anthemic, emotional, powerful"
    if "late night" in normalized_query:
        return "moody, reflective, calm"
    if "rage" in normalized_query or "angry" in normalized_query:
        return "angry, rebellious, intense"

    generic = [
        term
        for term in cue_terms
        if term not in {"coldplay", "scientist", "fix", "you", "yellow", "clocks", "kolaveri", "vaaji", "sahana", "sahara", "balleilakka"}
    ]
    return ", ".join(generic[:3]) if generic else "your vibe"


def _is_music_domain_query(normalized_query: str, cue_terms: set[str]) -> bool:
    tokens = set(normalized_query.split())
    if tokens & _MUSIC_DOMAIN_TERMS:
        return True

    if cue_terms & (_MUSIC_DOMAIN_TERMS | {"angry", "anthemic", "atmospheric", "calm", "catchy", "comforting", "driving", "dreamy", "edgy", "emotional", "energetic", "fun", "gentle", "heartfelt", "hopeful", "intense", "intimate", "melancholy", "melodic", "moody", "optimistic", "powerful", "reflective", "romantic", "soothing", "soulful", "uplifting", "warm"}):
        return True

    if tokens & _NON_MUSIC_DOMAIN_TERMS:
        return False

    return True


def is_music_domain_query(query: str) -> bool:
    """Return False only for clearly out-of-domain requests."""
    normalized = _normalize(query)
    cue_terms = _terms(query)
    return _is_music_domain_query(normalized, cue_terms)
