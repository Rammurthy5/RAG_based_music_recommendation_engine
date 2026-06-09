import asyncio

from langchain_core.runnables import RunnableLambda

from app.rag import chain as chain_module
from app.models.schemas import EvalMetrics
from app.rag.provider_factory import LLMProvider
from app.rag.vectorstore import RetrievedTrack


class FakeLLM(RunnableLambda):
    def __init__(self, response: str):
        super().__init__(lambda _: response)
        self._response = response

    def get_num_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


def test_get_recommendations_uses_selected_provider(monkeypatch):
    monkeypatch.setattr(chain_module, "weaviate_retry", lambda fn: fn)
    monkeypatch.setattr(
        chain_module,
        "search_tracks",
        lambda query: [
            RetrievedTrack(
                title="Night Drive",
                artist="Test Artist",
                album="Test Album",
                genres=["pop"],
                musicbrainz_id="track-1",
                distance=0.12,
            )
        ],
    )
    monkeypatch.setattr(chain_module, "llm_breaker", lambda fn: fn)
    monkeypatch.setattr(
        chain_module,
        "compute_eval_metrics",
        lambda **kwargs: EvalMetrics(
            faithfulness=0.91,
            answer_relevancy=0.82,
            context_recall=None,
            context_precision=None,
        ),
    )
    monkeypatch.setattr(
        chain_module,
        "build_llm",
        lambda: LLMProvider(
            provider="gemini",
            model="gemini-3.5-flash",
            llm=FakeLLM(
                """[
                    {
                        "title": "Night Drive",
                        "artist": "Test Artist",
                        "album": "Test Album",
                        "genre": ["pop"],
                        "reason": "Fits the mood."
                    }
                ]"""
            ),
        ),
    )

    result = asyncio.run(chain_module.get_recommendations("late night music", limit=1))

    assert result.metadata.source == "full_rag"
    assert result.metadata.provider == "gemini"
    assert result.metadata.model == "gemini-3.5-flash"
    assert result.metadata.eval_metrics.faithfulness == 0.91
    assert result.recommendations[0].title == "Night Drive"
    assert result.recommendations[0].track_id == "track-1"


def test_get_recommendations_falls_back_on_llm_failure(monkeypatch):
    monkeypatch.setattr(chain_module, "weaviate_retry", lambda fn: fn)
    monkeypatch.setattr(
        chain_module,
        "search_tracks",
        lambda query: [
            RetrievedTrack(
                title="Morning Light",
                artist="Test Artist",
                album="Test Album",
                genres=["indie"],
                musicbrainz_id="track-2",
                distance=0.18,
            )
        ],
    )
    monkeypatch.setattr(chain_module, "llm_breaker", lambda fn: fn)
    monkeypatch.setattr(
        chain_module,
        "compute_eval_metrics",
        lambda **kwargs: EvalMetrics(
            faithfulness=1.0,
            answer_relevancy=0.75,
            context_recall=None,
            context_precision=None,
        ),
    )
    monkeypatch.setattr(
        chain_module,
        "build_llm",
        lambda: LLMProvider(
            provider="openai",
            model="gpt-5.4-mini",
            llm=RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("rate limited"))),
        ),
    )

    result = asyncio.run(chain_module.get_recommendations("soft morning songs", limit=1))

    assert result.metadata.source == "retrieval_only"
    assert result.metadata.provider == ""
    assert result.recommendations[0].title == "Morning Light"
    assert result.recommendations[0].track_id == "track-2"


def test_get_recommendations_includes_eval_metrics(monkeypatch):
    monkeypatch.setattr(chain_module, "weaviate_retry", lambda fn: fn)
    monkeypatch.setattr(
        chain_module,
        "search_tracks",
        lambda query: [
            RetrievedTrack(
                title="Night Drive",
                artist="Test Artist",
                album="Test Album",
                genres=["pop"],
                musicbrainz_id="track-1",
                distance=0.12,
            )
        ],
    )
    monkeypatch.setattr(chain_module, "llm_breaker", lambda fn: fn)
    monkeypatch.setattr(chain_module, "load_eval_examples", lambda: [])
    monkeypatch.setattr(
        chain_module,
        "compute_eval_metrics",
        lambda **kwargs: EvalMetrics(
            faithfulness=0.88,
            answer_relevancy=0.77,
            context_recall=None,
            context_precision=None,
        ),
    )
    monkeypatch.setattr(
        chain_module,
        "build_llm",
        lambda: LLMProvider(
            provider="gemini",
            model="gemini-3.5-flash",
            llm=FakeLLM(
                """[
                    {
                        "title": "Night Drive",
                        "artist": "Test Artist",
                        "album": "Test Album",
                        "genre": ["pop"],
                        "reason": "Fits the mood."
                    }
                ]"""
            ),
        ),
    )

    result = asyncio.run(chain_module.get_recommendations("late night music", limit=1))

    assert result.metadata.eval_metrics.faithfulness == 0.88
    assert result.metadata.eval_metrics.answer_relevancy == 0.77


def test_get_recommendations_uses_static_fallback_when_retrieval_fails(monkeypatch):
    monkeypatch.setattr(
        chain_module,
        "weaviate_retry",
        lambda fn: lambda query: (_ for _ in ()).throw(RuntimeError("weaviate down")),
    )
    monkeypatch.setattr(
        chain_module,
        "compute_eval_metrics",
        lambda **kwargs: EvalMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=None,
            context_precision=None,
        ),
    )

    result = asyncio.run(chain_module.get_recommendations("anything", limit=2))

    assert result.metadata.source == "fallback_cache"
    assert result.metadata.provider == ""
    assert len(result.recommendations) == 2


def test_fallback_response_ranks_relevant_candidates(monkeypatch):
    monkeypatch.setattr(
        chain_module,
        "get_fallback_recommendations",
        lambda: [
            {
                "title": "Soft Glow",
                "artist": "Ambient Artist",
                "album": "Quiet Hours",
                "genre": ["ambient"],
                "reason": "Calm and spacious.",
            },
            {
                "title": "Viva la Vida",
                "artist": "Coldplay",
                "album": "Viva la Vida or Death and All His Friends",
                "genre": ["alternative rock"],
                "reason": "Epic and emotionally big.",
            },
        ],
    )

    response = chain_module._fallback_response("anthemic energetic songs with big feelings", 1, 12)

    assert response.recommendations[0].title == "Viva la Vida"


def test_parse_llm_output_recovers_partial_json_array():
    text = """[
      {
        "title": "Crank It Up",
        "artist": "Ashley Tisdale",
        "album": "Guilty Pleasure",
        "genre": "pop, teen pop",
        "reason": "Short and complete."
      },
      {
        "title": "A.T.L.",
        "artist": "M.I.A.",
        "album": "MATA",
        "genre": "hip hop, experimental hip hop",
        "reason": "This one gets cut off
    """

    parsed = chain_module._parse_llm_output(text)

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Crank It Up"
