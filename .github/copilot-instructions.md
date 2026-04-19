# Project Guidelines

## Overview

RAG-based music recommendation engine. Users describe moods/vibes in natural language → system returns song recommendations with explanations. Three-service architecture deployed via Docker Compose.

## Architecture

| Service | Language | Port | Role |
|---------|----------|------|------|
| `rag-service` | Python (FastAPI) | 8000 | RAG pipeline: embeddings, Weaviate retrieval, Claude generation |
| `api-gateway` | Go (chi router) | 8080 | Thin proxy: CORS, rate limiting, request validation, circuit breaker |
| `frontend` | TypeScript (Next.js) | 3000 | Search UI with recommendation cards |
| `weaviate` | — | 8085 (host) / 8080 (container), gRPC 50051 | Vector database |

**Data flow:** Frontend → Go gateway (`/api/recommend`) → Python RAG service (`/recommend`) → Weaviate retrieval + Claude generation → response back up the chain.

**Ingestion (offline):** MusicBrainz API (metadata) + Genius API (lyrics) → chunk → embed (HuggingFace `all-MiniLM-L6-v2`) → Weaviate collection `MusicRecommendations`.

## Build and Test

```bash
# Start all services
docker-compose up --build

# Run Python tests
cd rag-service && pytest

# Run Go tests
cd api-gateway && go test ./...

# Create Weaviate schema (first time only)
docker-compose exec rag-service python -m scripts.create_schema

# Run ingestion pipeline (after schema created)
docker-compose exec rag-service python -m scripts.ingest

# Frontend dev
cd frontend && npm run dev
```

## Code Style

### Python (`rag-service/`)
- Python 3.12, type hints on all function signatures
- Pydantic v2 models for request/response schemas
- LangChain v0.2+ with LCEL chains (not legacy `RetrievalQA`)
- Import order: stdlib → third-party → local (isort compatible)
- Async FastAPI endpoints where possible

### Go (`api-gateway/`)
- Go 1.22+, standard project layout: `cmd/`, `internal/`
- `chi` router for HTTP handling
- Errors returned, not panicked — wrap with `fmt.Errorf("context: %w", err)`
- No exported types/functions without purpose; keep the gateway thin

### TypeScript (`frontend/`)
- Next.js App Router (not Pages Router)
- Functional components with TypeScript interfaces for props
- Tailwind CSS for styling, no CSS modules
- Server components by default; `"use client"` only when needed

## Conventions

- **Environment variables:** All secrets and config via `.env` file (see `.env.example`). Never commit `.env`.
- **API keys required:** `ANTHROPIC_API_KEY`, `GENIUS_ACCESS_TOKEN` (MusicBrainz requires no API key, only a user-agent string).
- **Embedding model:** `all-MiniLM-L6-v2` (384 dims, cosine distance). Must match between ingestion and query time. Embedded locally (not via Weaviate vectorizer).
- **Lyrics copyright:** Store excerpts only (≤500 chars) in vector store. Link to Genius for full text. Do not reproduce full lyrics in API responses.
- **Docker networking:** Services reference each other by Docker Compose service name (e.g., `weaviate:8080`, `rag-service:8000`). Weaviate host port is 8085 to avoid conflict with Go gateway.
- **RAG chain pattern:** Always use LCEL pipe syntax (`retriever | prompt | llm | parser`), not deprecated chain classes.
- **Go gateway is a thin proxy:** No RAG logic here. Validate input, proxy to Python service, forward response.
- **Weaviate schema:** Pre-create collection with typed properties via `scripts/create_schema.py`. Schema is immutable — changes require delete + recreate + re-ingest.

## Resilience

- **Timeouts at every boundary:** Weaviate query 5s, LLM 15s, embedding 30s (cold) / 2s (warm). Go gateway → rag-service 35s.
- **Circuit breaker:** `pybreaker` around LLM calls (open after 5 failures, reset after 60s). `sony/gobreaker` in Go gateway.
- **Retries with jitter:** `tenacity` for idempotent calls only (Weaviate retrieval, embeddings). Never retry LLM generation.
- **Graceful degradation:** LLM fails → return retrieval-only results with local explanations (`source: "retrieval_only"`). Weaviate fails → return cached fallback playlists (`source: "fallback_cache"`).
- **Similarity threshold:** Default 0.65. Below threshold → fallback path.
- **Diversity:** Max 2 songs per artist unless user explicitly names one.

## Response Metadata

Every `/recommend` response includes a `metadata` block:
- `source`: `"full_rag"` | `"retrieval_only"` | `"fallback_cache"`
- `prompt_id`: Version of the prompt template used
- `model`: LLM model name (e.g., `"claude-sonnet-4-20250514"`)
- `rag_config`: `{top_k, chunk_size, similarity_threshold, embedding_model}`
- `cost`: `{input_tokens, output_tokens, total_cost_usd}`
- `latency_ms`: End-to-end request time

## Key Directories

```
rag-service/app/rag/           # Core RAG pipeline (chain, vectorstore, embeddings, prompts)
rag-service/app/resilience/    # Circuit breaker, retry decorators, fallback logic
rag-service/scripts/           # Data ingestion (MusicBrainz client, Genius fetcher, schema creation)
api-gateway/internal/          # Go handlers, middleware, RAG service client
frontend/src/components/       # React components (SearchBar, RecommendationCard)
```
