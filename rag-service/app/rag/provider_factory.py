"""Helpers for constructing the configured LLM provider.

This keeps provider selection in one place so the chain can stay focused on
retrieval, prompting, and parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from app.config import settings

ProviderName = Literal["gemini", "openai"]


@dataclass(frozen=True)
class LLMProvider:
    """Resolved provider details plus the instantiated chat model."""

    provider: ProviderName
    model: str
    llm: Any


def _normalize_provider(provider: str) -> ProviderName:
    normalized = provider.strip().lower()
    if normalized not in {"gemini", "openai"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return cast(ProviderName, normalized)


def build_llm(provider: str | None = None) -> LLMProvider:
    """Build the configured chat model and return its provider metadata."""
    selected_provider = _normalize_provider(provider or settings.llm_provider)

    if selected_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            timeout=settings.llm_timeout,
            temperature=0,
            max_output_tokens=3072,
        )
        return LLMProvider(
            provider="gemini",
            model=settings.gemini_model,
            llm=llm,
        )

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0,
        max_tokens=3072,
    )
    return LLMProvider(
        provider="openai",
        model=settings.openai_model,
        llm=llm,
    )
