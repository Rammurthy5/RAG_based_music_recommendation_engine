# System Architecture

## Overview

```text
Frontend (Next.js) -> API Gateway (Go/chi) -> RAG Service (FastAPI) -> Weaviate
                                             -> Gemini / OpenAI
```

| Service | Language | Port | Role |
|---------|----------|------|------|
| frontend | TypeScript (Next.js 15) | 3000 | Search UI and response display |
| api-gateway | Go (chi router) | 8080 | Proxy, CORS, rate limiting, circuit breaker |
| rag-service | Python (FastAPI) | 8000 | Retrieval, prompting, generation, eval metadata |
| weaviate | - | 8085 (host) / 8080 (container) | Vector database |

## Retrieval Flow

1. Normalize the user query into intent cues.
2. Expand the query with mood, genre, era, and artist hints.
3. Generate a query embedding and search Weaviate with `near_vector`.
4. Merge duplicate candidates and rerank them with heuristic metadata boosts.
5. Ground the LLM response in the retrieved tracks.
6. Backfill missing slots from retrieved candidates only.
7. Return recommendations with metadata for provider, source, cost, latency, and eval metrics.

## Resilience

- LLM failures fall back to retrieval-only results.
- Weaviate failures fall back to curated cached recommendations.
- Circuit breakers protect the Python service and the Go gateway.
- Retries are limited to idempotent retrieval and embedding calls.
- Timeouts are set per hop so slow dependencies do not stall the request path.

## Data Model

The Weaviate collection stores track-level records with:

- title
- artist
- album
- genres
- release_year
- lyrics_excerpt
- genius_url
- musicbrainz_id
- content

The embedded `content` field combines metadata, mood descriptors derived from genres, and a short lyric excerpt when available.

## Configuration

The main runtime settings are environment-driven through `rag-service/app/config.py` and `.env.example`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `gemini` | Selects Gemini or OpenAI |
| `GOOGLE_API_KEY` | - | Gemini auth |
| `OPENAI_API_KEY` | - | OpenAI auth |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model name |
| `OPENAI_MODEL` | `gpt-5.4-mini` | OpenAI model name |
| `TOP_K` | `3` | Retrieval result count |
| `SIMILARITY_THRESHOLD` | `0.20` | Minimum similarity filter |
| `WEAVIATE_HOST` | `weaviate` | Weaviate host |

## Project Layout

```text
rag-service/
  app/
    config.py
    models/schemas.py
    rag/
      chain.py
      vectorstore.py
      query_intent.py
      provider_factory.py
      evaluation.py
  scripts/
    create_schema.py
    ingest.py
    musicbrainz_client.py
    genius_fetcher.py
api-gateway/
frontend/
```

