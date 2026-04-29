"""Data ingestion pipeline.

Orchestrates:  MusicBrainz (metadata) + Genius (lyrics) → chunk → embed → Weaviate.

Usage:
    docker-compose exec rag-service python -m scripts.ingest
    docker-compose exec rag-service python -m scripts.ingest --per-seed 10  # smaller run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import weaviate
import weaviate.classes as wvc

from app.config import settings
from app.rag.embeddings import embed_texts
from scripts.create_schema import COLLECTION_NAME, create_schema
from scripts.genius_fetcher import fetch_lyrics
from scripts.musicbrainz_client import (
    DEFAULT_SEED_QUERIES,
    TrackMetadata,
    fetch_all_seeds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Batch size for embedding + Weaviate upserts
EMBED_BATCH_SIZE = 64


def build_content(track: TrackMetadata, lyrics_excerpt: str) -> str:
    """Create the text passage that will be embedded.

    Combines title, artist, album, genres, and a lyrics excerpt into a single
    string optimised for semantic search on moods/vibes.
    """
    parts: list[str] = [
        f'"{track.title}" by {track.artist}.',
    ]
    if track.album:
        parts.append(f"Album: {track.album}.")
    if track.genres:
        parts.append(f"Genres: {', '.join(track.genres)}.")
    if track.release_year:
        parts.append(f"Released: {track.release_year}.")
    if lyrics_excerpt:
        parts.append(f"Lyrics excerpt: {lyrics_excerpt}")
    return " ".join(parts)


def _connect_weaviate() -> weaviate.WeaviateClient:
    """Connect to Weaviate (using Docker-internal or host settings)."""
    client = weaviate.connect_to_custom(
        http_host=settings.weaviate_host,
        http_port=settings.weaviate_http_port,
        http_secure=False,
        grpc_host=settings.weaviate_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=False,
    )
    if not client.is_ready():
        logger.error("Weaviate is not ready.")
        sys.exit(1)
    return client


def ingest(
    per_seed: int = 25,
    skip_lyrics: bool = False,
) -> None:
    """Run the full ingestion pipeline."""
    start = time.perf_counter()

    # --- 1. Connect to Weaviate and ensure schema exists ---
    client = _connect_weaviate()
    try:
        create_schema(client)
        collection = client.collections.get(COLLECTION_NAME)

        # --- 2. Fetch metadata from MusicBrainz ---
        logger.info("Fetching metadata from MusicBrainz (per_seed=%d) …", per_seed)
        tracks = fetch_all_seeds(seeds=DEFAULT_SEED_QUERIES, per_seed=per_seed)
        if not tracks:
            logger.warning("No tracks fetched. Exiting.")
            return
        logger.info("Fetched %d tracks from MusicBrainz.", len(tracks))

        # --- 3. Fetch lyrics from Genius (optional) ---
        lyrics_map: dict[str, tuple[str, str]] = {}  # mb_id → (excerpt, url)
        if not skip_lyrics:
            logger.info("Fetching lyrics from Genius …")
            for i, track in enumerate(tracks):
                excerpt, url = fetch_lyrics(track.title, track.artist)
                lyrics_map[track.musicbrainz_id] = (excerpt, url)
                if (i + 1) % 10 == 0:
                    logger.info("  Genius progress: %d / %d", i + 1, len(tracks))
        else:
            logger.info("Skipping lyrics fetch (--skip-lyrics).")

        # --- 4. Build content strings ---
        contents: list[str] = []
        for track in tracks:
            excerpt, _ = lyrics_map.get(track.musicbrainz_id, ("", ""))
            contents.append(build_content(track, excerpt))

        # --- 5. Embed in batches ---
        logger.info("Embedding %d tracks …", len(contents))
        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(contents), EMBED_BATCH_SIZE):
            batch = contents[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectors = embed_texts(batch)
            all_vectors.extend(vectors)
            logger.info(
                "  Embedded %d / %d", len(all_vectors), len(contents)
            )

        # --- 6. Batch insert into Weaviate ---
        logger.info("Inserting %d objects into Weaviate …", len(tracks))
        with collection.batch.dynamic() as batch:
            for idx, track in enumerate(tracks):
                excerpt, url = lyrics_map.get(track.musicbrainz_id, ("", ""))
                properties = {
                    "title": track.title,
                    "artist": track.artist,
                    "album": track.album,
                    "genres": track.genres,
                    "release_year": track.release_year,
                    "lyrics_excerpt": excerpt,
                    "genius_url": url,
                    "musicbrainz_id": track.musicbrainz_id,
                    "content": contents[idx],
                }
                batch.add_object(properties=properties, vector=all_vectors[idx])

        obj_count = collection.aggregate.over_all(total_count=True).total_count
        elapsed = time.perf_counter() - start
        logger.info(
            "Ingestion complete: %d objects in collection, took %.1fs.",
            obj_count,
            elapsed,
        )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest music data into Weaviate")
    parser.add_argument(
        "--per-seed",
        type=int,
        default=25,
        help="Number of recordings per seed query (default: 25)",
    )
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Skip Genius lyrics fetch (ingest metadata only)",
    )
    args = parser.parse_args()
    ingest(per_seed=args.per_seed, skip_lyrics=args.skip_lyrics)


if __name__ == "__main__":
    main()
