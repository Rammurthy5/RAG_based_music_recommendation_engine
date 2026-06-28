from app.rag.query_intent import build_query_intent
from app.rag.vectorstore import RetrievedTrack, _merge_candidates, _rerank_tracks


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


def test_build_query_intent_expands_short_vibe_queries():
    intent = build_query_intent("rage")

    assert "rage" in intent.expanded_query
    assert "angry" in intent.expanded_query
    assert "rebellious" in intent.summary or "intent cues" in intent.summary


def test_merge_candidates_keeps_best_distance_per_track():
    worse = RetrievedTrack(
        title="Clocks",
        artist="Coldplay",
        musicbrainz_id="track-1",
        distance=0.6,
    )
    better = RetrievedTrack(
        title="Clocks",
        artist="Coldplay",
        musicbrainz_id="track-1",
        distance=0.2,
    )

    merged = _merge_candidates([worse, better])

    assert len(merged) == 1
    assert merged[0].distance == 0.2


def test_rerank_tracks_prefers_cross_encoder_matches_over_raw_similarity(monkeypatch):
    monkeypatch.setattr(
        "app.rag.vectorstore.get_cross_encoder",
        lambda: FakeCrossEncoder({"Rebel Yell": 0.95, "Soft Glow": 0.05}),
    )

    tracks = [
        RetrievedTrack(
            title="Soft Glow",
            artist="Ambient Artist",
            genres=["ambient"],
            content="calm spacious meditative",
            distance=0.05,
        ),
        RetrievedTrack(
            title="Rebel Yell",
            artist="Rock Artist",
            genres=["rock"],
            content="angry rebellious cathartic",
            distance=0.30,
        ),
    ]

    ranked = _rerank_tracks("rage", tracks)

    assert ranked[0].title == "Rebel Yell"


def test_rerank_tracks_falls_back_when_cross_encoder_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.rag.vectorstore.get_cross_encoder",
        lambda: (_ for _ in ()).throw(RuntimeError("cross-encoder unavailable")),
    )

    tracks = [
        RetrievedTrack(
            title="Soft Glow",
            artist="Ambient Artist",
            genres=["ambient"],
            content="calm spacious meditative",
            distance=0.05,
        ),
        RetrievedTrack(
            title="Rebel Yell",
            artist="Rock Artist",
            genres=["rock"],
            content="angry rebellious cathartic",
            distance=0.30,
        ),
    ]

    ranked = _rerank_tracks("rage", tracks)

    assert ranked[0].title == "Rebel Yell"
