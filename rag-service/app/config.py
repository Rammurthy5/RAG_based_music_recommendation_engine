from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    genius_access_token: str = ""
    weaviate_host: str = "weaviate"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # Timeouts (seconds)
    weaviate_query_timeout: int = 5
    llm_timeout: int = 15
    embedding_timeout: int = 30

    # Circuit breaker
    cb_fail_max: int = 5
    cb_reset_timeout: int = 60

    # RAG defaults
    top_k: int = 10
    similarity_threshold: float = 0.65

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
