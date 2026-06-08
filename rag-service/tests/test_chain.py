import asyncio

from langchain_core.runnables import RunnableLambda

from app.rag import chain as chain_module
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


def test_get_recommendations_uses_static_fallback_when_retrieval_fails(monkeypatch):
    monkeypatch.setattr(
        chain_module,
        "weaviate_retry",
        lambda fn: lambda query: (_ for _ in ()).throw(RuntimeError("weaviate down")),
    )

    result = asyncio.run(chain_module.get_recommendations("anything", limit=2))

    assert result.metadata.source == "fallback_cache"
    assert result.metadata.provider == ""
    assert len(result.recommendations) == 2


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
