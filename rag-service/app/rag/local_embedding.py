"""Deterministic offline embeddings used for eval and local fallbacks."""

from __future__ import annotations

import hashlib
import re

LOCAL_EMBED_DIM = 128


def local_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBED_DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % LOCAL_EMBED_DIM
        vector[bucket] += 1.0
    return vector
