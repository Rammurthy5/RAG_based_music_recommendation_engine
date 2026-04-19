import logging
import time

from fastapi import APIRouter

from app.models.schemas import RecommendRequest, RecommendResponse, ResponseMetadata

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Return music recommendations for a natural language mood/vibe query.

    Full RAG pipeline will be wired in Phase 3. Currently returns a stub.
    """
    start = time.perf_counter()

    # TODO: Phase 3 — invoke LCEL RAG chain
    # TODO: Phase 3B — circuit breaker, fallback, diversity, filters

    latency_ms = int((time.perf_counter() - start) * 1000)

    return RecommendResponse(
        query=request.query,
        recommendations=[],
        metadata=ResponseMetadata(
            source="full_rag",
            latency_ms=latency_ms,
        ),
    )
