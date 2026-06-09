"""Run an offline evaluation pass over the labeled eval set.

This uses the app's current recommendation pipeline and reports dataset-level
averages for the four evaluation metrics:
- faithfulness
- answer_relevancy
- context_recall
- context_precision
"""

from __future__ import annotations

import asyncio
import json
import os

from app.rag.evaluation import load_eval_examples, summarize_eval_metrics
from app.rag.chain import get_recommendations


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


async def _run() -> tuple[list[dict], dict[str, float | None]]:
    examples = load_eval_examples()
    rows: list[dict] = []
    per_query_metrics = []

    for example in examples:
        limit = max(1, len(example.reference_tracks))
        result = await get_recommendations(query=example.query, limit=limit)
        metrics = result.metadata.eval_metrics
        per_query_metrics.append(metrics)
        rows.append(
            {
                "query": example.query,
                "source": result.metadata.source,
                "faithfulness": metrics.faithfulness,
                "answer_relevancy": metrics.answer_relevancy,
                "context_recall": metrics.context_recall,
                "context_precision": metrics.context_precision,
            }
        )

    summary = summarize_eval_metrics(per_query_metrics)
    return rows, summary.model_dump()


def main() -> None:
    rows, summary = asyncio.run(_run())

    print("Overall evaluation summary")
    print(f"Examples: {len(rows)}")
    print(f"Faithfulness: {_fmt(summary['faithfulness'])}")
    print(f"Answer relevancy: {_fmt(summary['answer_relevancy'])}")
    print(f"Context recall: {_fmt(summary['context_recall'])}")
    print(f"Context precision: {_fmt(summary['context_precision'])}")
    print()
    print("Per-query detail")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
