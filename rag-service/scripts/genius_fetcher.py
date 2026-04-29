"""Genius metadata fetcher.

Given a song title + artist, searches the official Genius API (api.genius.com)
and returns the Genius URL.  Uses only the authenticated API — no web scraping.

Lyrics are NOT fetched here.  The lyricsgenius library scrapes genius.com HTML
pages which returns 403.  Instead we store only the Genius URL so users can
click through, and we use the song description snippet from the API as a
lightweight text supplement.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

GENIUS_SEARCH_URL = "https://api.genius.com/search"

_client: httpx.Client | None = None


def _get_client() -> httpx.Client | None:
    """Return a shared httpx client with the Genius auth header."""
    global _client
    if _client is not None:
        return _client
    token = settings.genius_access_token
    if not token:
        logger.warning("GENIUS_ACCESS_TOKEN not set — Genius data will be unavailable.")
        return None
    _client = httpx.Client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    return _client


def fetch_lyrics(title: str, artist: str) -> tuple[str, str]:
    """Search the official Genius API and return (description_snippet, genius_url).

    Returns ("", "") if the song is not found or the API is unavailable.
    The description_snippet is a short text blurb from Genius (not full lyrics).
    """
    client = _get_client()
    if client is None:
        return "", ""

    query = f"{title} {artist}"
    try:
        resp = client.get(GENIUS_SEARCH_URL, params={"q": query})
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Genius API error for '%s - %s': %s %s",
            artist, title, exc.response.status_code, exc.response.reason_phrase,
        )
        return "", ""
    except httpx.HTTPError as exc:
        logger.warning("Genius request failed for '%s - %s': %s", artist, title, exc)
        return "", ""

    data = resp.json()
    hits = data.get("response", {}).get("hits", [])
    if not hits:
        return "", ""

    # Take the first result
    song_info = hits[0].get("result", {})
    url = song_info.get("url", "")
    # Use the full_title as a snippet (e.g. "Sultans of Swing by Dire Straits")
    snippet = song_info.get("full_title", "")

    return snippet, url
