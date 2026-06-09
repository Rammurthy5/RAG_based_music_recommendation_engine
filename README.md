# 🎵 RAG Music Recommendation Engine

Describe a mood, vibe, or moment in natural language and get personalized song recommendations powered by Retrieval-Augmented Generation.

> **Retrieval stack:** query intent expansion -> Weaviate `near_vector` search -> duplicate merge -> heuristic rerank -> LLM grounded on retrieved tracks.  
> **RAGAS target:** Faithfulness `0.70+`, Answer Relevancy `0.70+`, Context Recall `0.70+`, Context Precision `0.70+`.

## Project Definition

This project is a full-stack music recommender built around a RAG service, an API gateway, and a Next.js frontend. The system turns a short natural-language prompt into track recommendations, keeps the response schema stable for clients, and records metadata such as provider, cost, latency, and evaluation metrics.

## Data Processing

The corpus is built from public music metadata and short lyric excerpts.

- **Sources:** MusicBrainz for track metadata and Genius for lyric snippets or URLs during ingestion.
- **Guardrails:** ingestion uses official APIs, rate limits, schema-controlled Weaviate writes, and prompt grounding so the LLM can only recommend tracks from retrieved context.
- **PII handling:** the codebase does not include a dedicated PII extraction or redaction pipeline. User queries are treated as transient request text, and the stored dataset is track-level music metadata rather than user profiles.

## Evals

Evaluation is task-specific and centered on recommendation quality rather than generic text generation.

- **Task-specific:** labeled eval queries map to reference tracks so retrieval and grounding can be checked against known targets.
- **Error handling:** when recommendations or references are missing, the metric code degrades gracefully instead of failing the request.
- **Cost:** token usage and estimated LLM cost are included in response metadata.
- **Latency:** end-to-end request latency is included in response metadata.
- **Metrics:** Faithfulness, Answer Relevancy, Context Recall, and Context Precision are computed and exposed through `metadata.eval_metrics`.

## Docs

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [Setup and Usage](./SETUP_AND_USAGE.md)
