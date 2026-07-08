from app.rag.tracing import build_trace_config, traced, tracing_enabled


def test_tracing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert not tracing_enabled()


def test_traceable_passthrough_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    @traced(name="example", run_type="chain")
    def sample(value: int) -> int:
        return value + 1

    assert sample(2) == 3


def test_build_trace_config_includes_request_context():
    config = build_trace_config(
        request_id="req-123",
        query="late night music",
        stage="recommendation",
        tags=["rag-service"],
    )

    assert config["metadata"]["request_id"] == "req-123"
    assert config["metadata"]["query"] == "late night music"
    assert config["metadata"]["stage"] == "recommendation"
    assert "request_id:req-123" in config["tags"]
    assert "rag-service" in config["tags"]
