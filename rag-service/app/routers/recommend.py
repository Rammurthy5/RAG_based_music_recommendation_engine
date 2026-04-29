import logging
import time

from fastapi import APIRouter

from app.models.schemas import RecommendRequest, RecommendResponse
from app.rag.chain import get_recommendations

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Return music recommendations for a natural language mood/vibe query."""
    return await get_recommendations(query=request.query, limit=request.limit)
