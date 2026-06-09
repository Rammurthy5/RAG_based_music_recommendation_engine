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

## Start the Stack

```bash
docker-compose up --build
```

The usual startup order is Weaviate, rag-service, api-gateway, then frontend.

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

