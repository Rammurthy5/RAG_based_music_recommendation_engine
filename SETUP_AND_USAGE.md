# Setup and Usage

## Prerequisites

- Docker and Docker Compose
- A Google API key for Gemini mode
- An OpenAI API key for OpenAI mode
- A Genius API token for ingestion

## Configure Environment

```bash
cp .env.example .env
```

Then set `LLM_PROVIDER` and the matching API key in `.env`.

Optional retrieval tuning:

- `HYBRID_ALPHA=0.5`
- `RETRIEVAL_CANDIDATE_MULTIPLIER=4`
- `RETRIEVAL_MIN_CANDIDATE_POOL=12`
- `CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `CROSS_ENCODER_BATCH_SIZE=16`
- `CROSS_ENCODER_MAX_CANDIDATES=30`

## Start the Stack

```bash
docker-compose up --build
```

The usual startup order is Weaviate, rag-service, api-gateway, then frontend.

The RAG service now uses Weaviate hybrid retrieval, then a local cross-encoder reranker, so exact-title/artist matches and semantic vibe matches are combined before ranking the final list.

## Create the Schema

```bash
docker-compose exec rag-service python -m scripts.create_schema
```

## Ingest Music Data

```bash
docker-compose exec rag-service python -m scripts.ingest
```

Optional smaller ingestion:

```bash
docker-compose exec rag-service python -m scripts.ingest --per-seed 10
```

## Open the App

Visit `http://localhost:3000` and enter a mood or vibe prompt.

## API

### `POST /api/recommend`

Request:

```json
{
  "query": "uplifting morning music with tropical vibes",
  "limit": 5
}
```

### `GET /api/health`

Liveness check.

### `GET /api/health/ready`

Readiness check for downstream dependencies.

## Run Services Individually

```bash
# RAG service
cd rag-service && uvicorn app.main:app --reload

# API gateway
cd api-gateway && go run ./cmd/server

# Frontend
cd frontend && npm run dev
```

## Run Tests

```bash
# Python
cd rag-service && pytest

# Go
cd api-gateway && go test ./...
```

## Evaluation Script

```bash
cd rag-service && python -m scripts.evaluate
```
