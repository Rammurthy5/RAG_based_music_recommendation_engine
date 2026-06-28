from __future__ import annotations

from types import SimpleNamespace

from app.rag import vectorstore
from app.rag.vectorstore import RetrievedTrack, _merge_candidates, search_tracks


class FakeObject:
    def __init__(self, properties: dict, *, distance: float | None = None, score: float | None = None):
        self.properties = properties
        self.metadata = SimpleNamespace(distance=distance, score=score)


class FakeResult:
    def __init__(self, objects: list[FakeObject]):
        self.objects = objects


class FakeCrossEncoder:
    def __init__(self, scores_by_fragment: dict[str, float]):
        self.scores_by_fragment = scores_by_fragment

    def predict(self, pairs, batch_size=None, show_progress_bar=None):
        scores = []
        for _, candidate_text in pairs:
            score = 0.0
            for fragment, fragment_score in self.scores_by_fragment.items():
                if fragment.lower() in candidate_text.lower():
                    score = fragment_score
                    break
            scores.append(score)
        return scores


class FakeQuery:
    def __init__(self):
        self.hybrid_calls: list[dict] = []
        self.near_vector_calls: list[dict] = []
        self.hybrid_results: dict[str, list[FakeObject]] = {}
        self.vector_results: dict[str, list[FakeObject]] = {}
        self.hybrid_error: Exception | None = None

    def hybrid(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        if self.hybrid_error is not None:
            raise self.hybrid_error
        return FakeResult(self.hybrid_results.get(kwargs["query"], []))

    def near_vector(self, **kwargs):
        self.near_vector_calls.append(kwargs)
        return FakeResult(self.vector_results.get(kwargs["limit"], self.vector_results.get("default", [])))


class FakeCollection:
    def __init__(self, query: FakeQuery):
        self.query = query


class FakeCollections:
    def __init__(self, collection: FakeCollection):
        self._collection = collection

    def get(self, _name: str):
        return self._collection


class FakeClient:
    def __init__(self, collection: FakeCollection):
        self.collections = FakeCollections(collection)


def _make_client() -> tuple[FakeClient, FakeQuery]:
    query = FakeQuery()
    collection = FakeCollection(query)
    return FakeClient(collection), query


def test_search_tracks_uses_hybrid_for_original_and_expanded_variants_and_candidate_pool(monkeypatch):
    client, query = _make_client()
    query.hybrid_results = {
        "rage": [
            FakeObject(
                {
                    "title": "Rebel Yell",
                    "artist": "Rock Artist",
                    "genres": ["rock"],
                    "musicbrainz_id": "track-1",
                },
                score=0.72,
            )
        ],
        "rage angry intense aggressive rebellious cathartic powerful": [
            FakeObject(
                {
                    "title": "Rebel Yell",
                    "artist": "Rock Artist",
                    "genres": ["rock"],
                    "musicbrainz_id": "track-1",
                },
                score=0.93,
            ),
            FakeObject(
                {
                    "title": "Angry Anthem",
                    "artist": "Rock Artist",
                    "genres": ["rock"],
                    "musicbrainz_id": "track-2",
                },
                score=0.6,
            ),
        ],
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    monkeypatch.setattr(vectorstore, "embed_single", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(vectorstore, "embedding_retry", lambda fn: fn)
    monkeypatch.setattr(
        vectorstore,
        "get_cross_encoder",
        lambda: FakeCrossEncoder({"Rebel Yell": 0.92, "Angry Anthem": 0.84}),
    )
    monkeypatch.setattr(vectorstore.settings, "top_k", 3)
    monkeypatch.setattr(vectorstore.settings, "retrieval_candidate_multiplier", 4)
    monkeypatch.setattr(vectorstore.settings, "retrieval_min_candidate_pool", 12)

    tracks = search_tracks("rage", top_k=3)

    assert len(query.hybrid_calls) == 2
    assert query.hybrid_calls[0]["limit"] == 12
    assert query.hybrid_calls[1]["limit"] == 12
    assert query.hybrid_calls[0]["query_properties"] == vectorstore.HYBRID_QUERY_PROPERTIES
    assert {call["query"] for call in query.hybrid_calls} == {
        "rage",
        "rage angry intense aggressive rebellious cathartic powerful",
    }
    assert [track.title for track in tracks] == ["Rebel Yell", "Angry Anthem"]


def test_search_tracks_falls_back_to_vector_only_when_hybrid_errors(monkeypatch):
    client, query = _make_client()
    query.hybrid_error = RuntimeError("hybrid unavailable")
    query.vector_results = {
        "default": [
            FakeObject(
                {
                    "title": "Morning Light",
                    "artist": "Test Artist",
                    "genres": ["indie"],
                    "musicbrainz_id": "track-2",
                },
                distance=0.18,
            )
        ]
    }
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    monkeypatch.setattr(vectorstore, "embed_single", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(vectorstore, "embedding_retry", lambda fn: fn)
    monkeypatch.setattr(
        vectorstore,
        "get_cross_encoder",
        lambda: FakeCrossEncoder({"Morning Light": 0.87}),
    )

    tracks = search_tracks("soft morning songs", top_k=1)

    assert len(query.hybrid_calls) == 1
    assert len(query.near_vector_calls) >= 1
    assert tracks[0].title == "Morning Light"


def test_hybrid_search_improves_exact_title_recall_over_vector_only(monkeypatch):
    hybrid_client, hybrid_query = _make_client()
    hybrid_query.hybrid_results = {
        "Clocks": [
            FakeObject(
                {
                    "title": "Clocks",
                    "artist": "Coldplay",
                    "genres": ["alternative rock"],
                    "musicbrainz_id": "track-clocks",
                },
                score=0.91,
            ),
            FakeObject(
                {
                    "title": "Soft Glow",
                    "artist": "Ambient Artist",
                    "genres": ["ambient"],
                    "musicbrainz_id": "track-soft",
                },
                score=0.2,
            ),
        ]
    }

    vector_client, vector_query = _make_client()
    vector_query.hybrid_error = RuntimeError("hybrid unavailable")
    vector_query.vector_results = {
        "default": [
            FakeObject(
                {
                    "title": "Soft Glow",
                    "artist": "Ambient Artist",
                    "genres": ["ambient"],
                    "musicbrainz_id": "track-soft",
                },
                distance=0.15,
            )
        ]
    }

    monkeypatch.setattr(vectorstore, "embed_single", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(vectorstore, "embedding_retry", lambda fn: fn)
    monkeypatch.setattr(
        vectorstore,
        "get_cross_encoder",
        lambda: FakeCrossEncoder({"Clocks": 0.96, "Soft Glow": 0.11}),
    )

    monkeypatch.setattr(vectorstore, "get_client", lambda: hybrid_client)
    hybrid_tracks = search_tracks("Clocks", top_k=1)

    monkeypatch.setattr(vectorstore, "get_client", lambda: vector_client)
    vector_tracks = search_tracks("Clocks", top_k=1)

    assert hybrid_tracks[0].title == "Clocks"
    assert vector_tracks[0].title == "Soft Glow"


def test_merge_candidates_prefers_higher_retrieval_score():
    worse = RetrievedTrack(
        title="Clocks",
        artist="Coldplay",
        musicbrainz_id="track-1",
        distance=0.6,
        retrieval_score=0.2,
    )
    better = RetrievedTrack(
        title="Clocks",
        artist="Coldplay",
        musicbrainz_id="track-1",
        distance=0.2,
        retrieval_score=0.8,
    )

    merged = _merge_candidates([worse, better])

    assert len(merged) == 1
    assert merged[0].retrieval_score == 0.8
