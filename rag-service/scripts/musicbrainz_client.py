"""MusicBrainz metadata fetcher.

Searches MusicBrainz for recordings by genre/tag seed queries and returns
structured metadata (title, artist, album, genres, release year).
MusicBrainz API requires no key — only a user-agent string.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import musicbrainzngs

logger = logging.getLogger(__name__)

# MusicBrainz requires a user-agent identifying the application
musicbrainzngs.set_useragent(
    "RAGMusicRecommendationEngine",
    "1.0",
    "https://github.com/rsi03/RAG_based_music_recommendation_engine",
)

# Respect MusicBrainz rate limit: 1 request per second
_MIN_REQUEST_INTERVAL = 1.1
_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Block until the MusicBrainz rate-limit window has passed."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


@dataclass
class TrackMetadata:
    title: str
    artist: str
    album: str = ""
    genres: list[str] = field(default_factory=list)
    release_year: int = 0
    musicbrainz_id: str = ""


# Seed queries spanning a variety of moods and genres
DEFAULT_SEED_QUERIES: list[dict[str, str]] = [
    # Genre tags
    {"tag": "rock"},
    {"tag": "pop"},
    {"tag": "jazz"},
    {"tag": "hip hop"},
    {"tag": "electronic"},
    {"tag": "classical"},
    {"tag": "r&b"},
    {"tag": "soul"},
    {"tag": "blues"},
    {"tag": "country"},
    {"tag": "folk"},
    {"tag": "metal"},
    {"tag": "reggae"},
    {"tag": "punk"},
    {"tag": "indie"},
    {"tag": "alternative"},
    {"tag": "latin"},
    {"tag": "funk"},
    {"tag": "ambient"},
    {"tag": "world"},
]


def _extract_tags(recording: dict) -> list[str]:
    """Extract tag names from a MusicBrainz recording result."""
    tags: list[str] = []
    for tag_entry in recording.get("tag-list", []):
        name = tag_entry.get("name", "").strip()
        if name:
            tags.append(name)
    return tags


def _extract_artist(recording: dict) -> str:
    """Extract the primary artist name from a recording."""
    credit_list = recording.get("artist-credit", [])
    if credit_list:
        artist_info = credit_list[0]
        if isinstance(artist_info, dict) and "artist" in artist_info:
            return artist_info["artist"].get("name", "Unknown Artist")
    return "Unknown Artist"


def _extract_release_info(recording: dict) -> tuple[str, int]:
    """Return (album_name, release_year) from the first release in a recording."""
    release_list = recording.get("release-list", [])
    if not release_list:
        return "", 0
    release = release_list[0]
    album = release.get("title", "")
    date_str = release.get("date", "")
    year = 0
    if date_str:
        try:
            year = int(date_str[:4])
        except (ValueError, IndexError):
            pass
    return album, year


def search_recordings_by_tag(
    tag: str, limit: int = 25, offset: int = 0
) -> list[TrackMetadata]:
    """Search MusicBrainz for recordings tagged with *tag*.

    Returns up to *limit* TrackMetadata objects.
    """
    _rate_limit()
    try:
        result = musicbrainzngs.search_recordings(
            tag=tag,
            limit=limit,
            offset=offset,
        )
    except musicbrainzngs.WebServiceError as exc:
        logger.warning("MusicBrainz search failed for tag '%s': %s", tag, exc)
        return []

    tracks: list[TrackMetadata] = []
    for rec in result.get("recording-list", []):
        title = rec.get("title", "").strip()
        if not title:
            continue
        artist = _extract_artist(rec)
        album, year = _extract_release_info(rec)
        genres = _extract_tags(rec)
        mb_id = rec.get("id", "")

        tracks.append(
            TrackMetadata(
                title=title,
                artist=artist,
                album=album,
                genres=genres,
                release_year=year,
                musicbrainz_id=mb_id,
            )
        )
    return tracks


def fetch_all_seeds(
    seeds: list[dict[str, str]] | None = None,
    per_seed: int = 25,
) -> list[TrackMetadata]:
    """Iterate over seed queries and collect unique tracks (by MusicBrainz ID).

    Duplicates (same musicbrainz_id) are removed.
    """
    if seeds is None:
        seeds = DEFAULT_SEED_QUERIES

    seen_ids: set[str] = set()
    all_tracks: list[TrackMetadata] = []

    for seed in seeds:
        tag = seed.get("tag", "")
        if not tag:
            continue
        logger.info("Fetching MusicBrainz recordings for tag: %s", tag)
        tracks = search_recordings_by_tag(tag, limit=per_seed)
        for t in tracks:
            if t.musicbrainz_id and t.musicbrainz_id in seen_ids:
                continue
            if t.musicbrainz_id:
                seen_ids.add(t.musicbrainz_id)
            all_tracks.append(t)

    logger.info("Fetched %d unique tracks from MusicBrainz.", len(all_tracks))
    return all_tracks
