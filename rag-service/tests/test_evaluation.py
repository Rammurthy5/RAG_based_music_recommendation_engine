from app.models.schemas import Recommendation
from app.rag.evaluation import (
    EvalMetrics,
    TrackLabel,
    compute_eval_metrics,
    load_eval_examples,
    summarize_eval_metrics,
)
from app.rag.vectorstore import RetrievedTrack


def fake_embed(text: str) -> list[float]:
    text = text.lower()
    if "late night" in text or "night drive" in text or "test artist" in text:
        return [1.0, 0.0]
    return [0.0, 1.0]


def test_compute_eval_metrics_scores_grounded_recommendations():
    retrieved = [
        RetrievedTrack(
            title="Night Drive",
            artist="Test Artist",
            album="Test Album",
            genres=["pop"],
            musicbrainz_id="track-1",
            distance=0.12,
        ),
        RetrievedTrack(
            title="Morning Light",
            artist="Other Artist",
            album="Other Album",
            genres=["indie"],
            musicbrainz_id="track-2",
            distance=0.2,
        ),
    ]
    recommendations = [
        Recommendation(
            title="Night Drive",
            artist="Test Artist",
            album="Test Album",
            genre=["pop"],
            reason="Fits the mood.",
        )
    ]
    labels = [TrackLabel(title="Night Drive", artist="Test Artist")]

    metrics = compute_eval_metrics(
        query="late night music",
        retrieved_tracks=retrieved,
        recommendations=recommendations,
        reference_tracks=labels,
        embed_fn=fake_embed,
    )

    assert metrics.faithfulness == 1.0
    assert metrics.answer_relevancy == 1.0
    assert metrics.context_recall == 1.0
    assert metrics.context_precision == 0.5


def test_compute_eval_metrics_handles_missing_recommendations():
    metrics = compute_eval_metrics(
        query="anything",
        retrieved_tracks=[],
        recommendations=[],
        reference_tracks=None,
        embed_fn=fake_embed,
    )

    assert metrics.faithfulness == 0.0
    assert metrics.answer_relevancy == 0.0
    assert metrics.context_recall is None
    assert metrics.context_precision is None


def test_load_eval_examples_reads_eval_set():
    examples = load_eval_examples()

    assert len(examples) == 3
    assert examples[0].query == "comforting songs for a rough night"
    assert examples[0].reference_tracks[0].title == "The Scientist (acoustic)"


def test_summarize_eval_metrics_averages_values():
    summary = summarize_eval_metrics(
        [
            EvalMetrics(
                faithfulness=1.0,
                answer_relevancy=0.5,
                context_recall=1.0,
                context_precision=0.25,
            ),
            EvalMetrics(
                faithfulness=0.5,
                answer_relevancy=1.0,
                context_recall=None,
                context_precision=0.75,
            ),
        ]
    )

    assert summary.faithfulness == 0.75
    assert summary.answer_relevancy == 0.75
    assert summary.context_recall == 1.0
    assert summary.context_precision == 0.5
