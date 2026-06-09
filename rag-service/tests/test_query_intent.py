from app.rag.query_intent import build_query_intent
from app.rag.vectorstore import RetrievedTrack, _merge_candidates, _rerank_tracks


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


def test_rerank_tracks_prefers_intent_matches_over_raw_similarity():
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
