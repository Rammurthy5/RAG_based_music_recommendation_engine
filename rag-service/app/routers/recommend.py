from fastapi import APIRouter, Request

from app.models.schemas import RecommendRequest, RecommendResponse
from app.rag.chain import get_recommendations
from app.rag.tracing import traced

router = APIRouter()


@traced(name="recommend_route", run_type="chain")
async def _recommend_route(
    query: str,
    limit: int,
    request_id: str | None = None,
) -> RecommendResponse:
    return await get_recommendations(query=query, limit=limit, request_id=request_id)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    """Return music recommendations for a natural language mood/vibe query."""
    request_id = getattr(request.state, "request_id", None)
    return await _recommend_route(
        query=payload.query,
        limit=payload.limit,
        request_id=request_id,
    )
