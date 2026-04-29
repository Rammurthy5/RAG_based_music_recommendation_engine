# 🎵 RAG Music Recommendation Engine

Describe a mood, vibe, or moment in natural language — get personalized song recommendations powered by Retrieval-Augmented Generation.

## Architecture

```
┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Frontend  │────▶│  API Gateway  │────▶│  RAG Service  │────▶│ Weaviate │
│  (Next.js) │◀────│   (Go/chi)   │◀────│  (FastAPI)    │◀────│ (Vector) │
│   :3000    │     │    :8080      │     │    :8000      │     │  :8085   │
└───────────┘     └──────────────┘     └──────────────┘     └──────────┘
                                              │
                                              ▼
                                        Claude (LLM)
```

| Service | Language | Port | Role |
|---------|----------|------|------|
| **frontend** | TypeScript (Next.js 15) | 3000 | Search UI with recommendation cards |
| **api-gateway** | Go (chi router) | 8080 | Proxy, CORS, rate limiting, circuit breaker |
| **rag-service** | Python (FastAPI) | 8000 | RAG pipeline: embed → retrieve → generate |
| **weaviate** | — | 8085 (host) / 8080 (container) | Vector database |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)
- A [Genius API token](https://genius.com/api-clients) (for ingestion only)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and GENIUS_ACCESS_TOKEN
```

### 2. Start all services

```bash
docker-compose up --build
```

Wait for all health checks to pass. You'll see the services start in dependency order:
Weaviate → rag-service → api-gateway → frontend.

### 3. Create the Weaviate schema (first time only)

```bash
docker-compose exec rag-service python -m scripts.create_schema
```

### 4. Ingest music data

```bash
docker-compose exec rag-service python -m scripts.ingest
```

This fetches metadata from MusicBrainz and lyrics excerpts from Genius, chunks and embeds them, then stores vectors in Weaviate.

### 5. Open the app

Visit **http://localhost:3000** and describe a vibe!

## API Reference

### `POST /api/recommend`

Gateway endpoint that proxies to the RAG service.

**Request:**
```json
{
  "query": "uplifting morning music with tropical vibes",
  "limit": 5
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `query` | string | yes | 1–500 characters |
| `limit` | integer | no | 1–20 (default: 5) |

**Response:**
```json
{
  "query": "uplifting morning music with tropical vibes",
  "recommendations": [
    {
      "title": "Walking on Sunshine",
      "artist": "Katrina & The Waves",
      "album": "Walking on Sunshine",
      "genre": ["pop", "dance"],
      "reason": "Captures the uplifting energy and tropical feel you described.",
      "artwork_url": "",
      "track_id": "abc123",
      "similarity_score": 0.87
    }
  ],
  "metadata": {
    "source": "full_rag",
    "prompt_id": "v1-default",
    "model": "claude-sonnet-4-20250514",
    "rag_config": {
      "top_k": 10,
      "chunk_size": 500,
      "similarity_threshold": 0.65,
      "embedding_model": "all-MiniLM-L6-v2"
    },
    "cost": {
      "input_tokens": 450,
      "output_tokens": 120,
      "total_cost_usd": 0.00231
    },
    "latency_ms": 1240
  }
}
```

### `GET /api/health`

Liveness check. Returns `{"status": "ok"}`.

### `GET /api/health/ready`

Deep readiness check including Weaviate connectivity and circuit breaker state.

## Resilience

The system degrades gracefully through three tiers:

| Source | Meaning | Trigger |
|--------|---------|---------|
| `full_rag` | Retrieval + LLM generation | Normal operation |
| `retrieval_only` | Weaviate results, no LLM | Claude fails or circuit breaker open |
| `fallback_cache` | Static curated playlists | Weaviate unavailable |

**Circuit breakers:** `pybreaker` wraps LLM calls (opens after 5 failures, resets after 60s). `gobreaker` in the Go gateway wraps rag-service calls.

**Rate limiting:** 20 req/s with burst of 40 at the gateway.

**Timeouts:** Weaviate 5s, LLM 15s, embedding 30s (cold) / 2s (warm), gateway → rag-service 35s.

## Development

### Run services individually

```bash
# Python RAG service
cd rag-service && pip install -r requirements.txt && uvicorn app.main:app --reload

# Go gateway
cd api-gateway && go run ./cmd/server

# Frontend
cd frontend && npm run dev
```

### Run tests

```bash
# Python
cd rag-service && pytest

# Go
cd api-gateway && go test ./...
```

## Project Structure

```
├── docker-compose.yml
├── .env.example
├── rag-service/
│   ├── app/
│   │   ├── main.py              # FastAPI app, health checks, middleware
│   │   ├── config.py            # Pydantic settings
│   │   ├── models/schemas.py    # Request/response Pydantic models
│   │   ├── rag/
│   │   │   ├── chain.py         # LCEL chain: retrieve → prompt → LLM → parse
│   │   │   ├── vectorstore.py   # Weaviate nearVector search
│   │   │   ├── embeddings.py    # HuggingFace all-MiniLM-L6-v2
│   │   │   └── prompts.py       # Prompt templates
│   │   ├── resilience/          # Circuit breaker, retry decorators, fallbacks
│   │   └── routers/recommend.py # POST /recommend endpoint
│   └── scripts/
│       ├── create_schema.py     # Weaviate collection setup
│       ├── ingest.py            # Ingestion orchestrator
│       ├── musicbrainz_client.py# MusicBrainz metadata fetcher
│       └── genius_fetcher.py    # Genius lyrics fetcher
├── api-gateway/
│   ├── cmd/server/main.go       # Entrypoint, middleware stack, routes
│   ├── config/config.go         # Environment-based configuration
│   └── internal/
│       ├── client/rag_client.go # HTTP client with circuit breaker
│       ├── handler/handler.go   # Request validation, proxy, health checks
│       └── middleware/          # Rate limiter, logger, security headers, etc.
└── frontend/
    └── src/
        ├── app/
        │   ├── layout.tsx       # Root layout
        │   ├── page.tsx         # Main page with search state management
        │   └── globals.css      # Tailwind base styles
        ├── components/
        │   ├── SearchBar.tsx    # Query input form
        │   ├── RecommendationCard.tsx  # Individual song card
        │   ├── RecommendationList.tsx  # Card list container
        │   └── MetadataBar.tsx  # Response source, latency, cost display
        └── lib/
            ├── types.ts         # TypeScript interfaces matching API contract
            └── api.ts           # API client with abort support
```

## Configuration

All configuration is via environment variables. See [.env.example](.env.example) for the full list.

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | rag-service | — | **Required.** Claude API key |
| `GENIUS_ACCESS_TOKEN` | rag-service | — | Required for ingestion only |
| `WEAVIATE_HOST` | rag-service | `weaviate` | Weaviate hostname |
| `RAG_SERVICE_URL` | api-gateway | `http://rag-service:8000` | RAG service URL |
| `PORT` | api-gateway | `8080` | Gateway listen port |
| `RATE_LIMIT` | api-gateway | `20` | Requests per second |
| `TOP_K` | rag-service | `10` | Number of vectors to retrieve |
| `SIMILARITY_THRESHOLD` | rag-service | `0.65` | Minimum cosine similarity |
