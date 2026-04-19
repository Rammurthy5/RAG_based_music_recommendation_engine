from typing import Literal

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)
    filters: dict | None = None


class Recommendation(BaseModel):
    title: str
    artist: str
    album: str = ""
    genre: list[str] = []
    reason: str = ""
    artwork_url: str = ""
    track_id: str = ""
    similarity_score: float = 0.0


class CostInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0


class RAGConfigInfo(BaseModel):
    top_k: int = 10
    chunk_size: int = 500
    similarity_threshold: float = 0.65
    embedding_model: str = "all-MiniLM-L6-v2"


class ResponseMetadata(BaseModel):
    source: Literal["full_rag", "retrieval_only", "fallback_cache"] = "full_rag"
    prompt_id: str = ""
    model: str = ""
    rag_config: RAGConfigInfo = RAGConfigInfo()
    cost: CostInfo = CostInfo()
    latency_ms: int = 0


class RecommendResponse(BaseModel):
    query: str
    recommendations: list[Recommendation] = []
    metadata: ResponseMetadata = ResponseMetadata()
