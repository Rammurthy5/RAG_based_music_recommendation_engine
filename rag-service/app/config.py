from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: Literal["gemini", "openai"] = "gemini"
    google_api_key: str = ""
    openai_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    openai_model: str = "gpt-5.4-mini"
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
    top_k: int = 3
    similarity_threshold: float = 0.20
    hybrid_alpha: float = 0.5
    retrieval_candidate_multiplier: int = 4
    retrieval_min_candidate_pool: int = 12
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 16
    cross_encoder_max_candidates: int = 30

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
