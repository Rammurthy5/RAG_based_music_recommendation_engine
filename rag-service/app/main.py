import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import recommend

logger = logging.getLogger(__name__)

# --- Structured logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Default factory for log records without request_id
_original_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _original_factory(*args, **kwargs)
    if not hasattr(record, "request_id"):
        record.request_id = "-"
    return record


logging.setLogRecordFactory(_record_factory)


# --- Lifespan: manage Weaviate client lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("rag-service starting up")
    yield
    logger.info("rag-service shutting down — closing Weaviate client")
    from app.rag.vectorstore import close_client
    close_client()


app = FastAPI(
    title="Music Recommendation RAG Service",
    version="1.0.0",
    lifespan=lifespan,
)

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
async def request_id_middleware(request: Request, call_next):
    """Propagate or generate X-Request-ID for distributed tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4().hex[:16]))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Latency-Ms"] = str(elapsed_ms)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — log details, return safe 500."""
    request_id = getattr(request.state, "request_id", "-")
    logger.error(
        "Unhandled exception: %s",
        exc,
        extra={"request_id": request_id},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error"},
        headers={"X-Request-ID": request_id},
    )
