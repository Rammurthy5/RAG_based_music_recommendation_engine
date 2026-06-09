from types import ModuleType

import sys

from app.config import settings
from app.rag.provider_factory import build_llm


class FakeGeminiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_llm_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash")
    monkeypatch.setattr(settings, "google_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "llm_timeout", 17)

    fake_module = ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeGeminiClient
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    bundle = build_llm()

    assert bundle.provider == "gemini"
    assert bundle.model == "gemini-3.5-flash"
    assert isinstance(bundle.llm, FakeGeminiClient)
    assert bundle.llm.kwargs["google_api_key"] == "test-gemini-key"
    assert bundle.llm.kwargs["timeout"] == 17
    assert bundle.llm.kwargs["temperature"] == 0


def test_build_llm_openai(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "gpt-5.4-mini")
    monkeypatch.setattr(settings, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(settings, "llm_timeout", 21)

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    bundle = build_llm()

    assert bundle.provider == "openai"
    assert bundle.model == "gpt-5.4-mini"
    assert isinstance(bundle.llm, FakeOpenAIClient)
    assert bundle.llm.kwargs["api_key"] == "test-openai-key"
    assert bundle.llm.kwargs["timeout"] == 21
    assert bundle.llm.kwargs["temperature"] == 0
