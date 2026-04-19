import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import recommend

app = FastAPI(title="Music Recommendation RAG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)


@app.get("/health")
async def health() -> dict:
    """Liveness + Weaviate connectivity check."""
    weaviate_status = "disconnected"
    try:
        from app.rag.vectorstore import get_client

        client = get_client()
        if client.is_ready():
            weaviate_status = "connected"
    except Exception:
        pass
    return {"status": "ok", "weaviate": weaviate_status}


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Latency-Ms"] = str(elapsed_ms)
    return response
