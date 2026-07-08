"""Optional LangSmith tracing helpers.

The helpers in this module keep LangSmith integration opt-in. When the tracing
environment variables are not configured, the decorator and config helpers are
safe no-ops so local development and CI keep behaving as before.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from dotenv import load_dotenv

P = ParamSpec("P")
R = TypeVar("R")

_TRUE_VALUES = {"1", "true", "yes", "on"}

_REPO_ENV = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_REPO_ENV, override=False)


def _copy_env_alias(source: str, target: str) -> None:
    if os.environ.get(target):
        return
    value = os.environ.get(source)
    if value:
        os.environ[target] = value


_copy_env_alias("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
_copy_env_alias("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
_copy_env_alias("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT")
_copy_env_alias("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT")

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover - optional dependency
    _langsmith_traceable = None


def tracing_enabled() -> bool:
    """Return True when LangSmith tracing should be active."""
    tracing_flag = (
        os.environ.get("LANGSMITH_TRACING", "").strip().lower() in _TRUE_VALUES
        or os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in _TRUE_VALUES
    )
    return tracing_flag and _langsmith_traceable is not None


def traced(
    *,
    name: str | None = None,
    run_type: str | None = None,
    tags: Iterable[str] | None = None,
):
    """Return a LangSmith traceable decorator when tracing is enabled."""
    if not tracing_enabled():
        def passthrough(fn: Callable[P, R]) -> Callable[P, R]:
            return fn

        return passthrough

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if run_type is not None:
        kwargs["run_type"] = run_type
    if tags is not None:
        kwargs["tags"] = list(tags)
    return _langsmith_traceable(**kwargs)


def build_trace_config(
    *,
    request_id: str | None = None,
    query: str | None = None,
    stage: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build LCEL config metadata so LangSmith traces can be correlated."""
    metadata: dict[str, Any] = {}
    if request_id:
        metadata["request_id"] = request_id
    if query:
        metadata["query"] = query
    if stage:
        metadata["stage"] = stage

    trace_tags = list(tags or [])
    if request_id:
        trace_tags.append(f"request_id:{request_id}")

    config: dict[str, Any] = {}
    if metadata:
        config["metadata"] = metadata
    if trace_tags:
        config["tags"] = trace_tags
    return config
