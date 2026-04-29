import { ResponseMetadata } from "@/lib/types";

interface MetadataBarProps {
  metadata: ResponseMetadata;
}

const SOURCE_LABELS: Record<ResponseMetadata["source"], string> = {
  full_rag: "Full RAG",
  retrieval_only: "Retrieval Only",
  fallback_cache: "Fallback Cache",
};

export function MetadataBar({ metadata }: MetadataBarProps) {
  const { source, model, latency_ms, cost } = metadata;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500 mt-2 px-1">
      <span
        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ${
          source === "full_rag"
            ? "bg-green-900/40 text-green-400"
            : source === "retrieval_only"
              ? "bg-yellow-900/40 text-yellow-400"
              : "bg-red-900/40 text-red-400"
        }`}
      >
        {SOURCE_LABELS[source]}
      </span>
      {model && <span>Model: {model}</span>}
      {latency_ms > 0 && <span>{(latency_ms / 1000).toFixed(1)}s</span>}
      {cost.total_cost_usd > 0 && (
        <span>${cost.total_cost_usd.toFixed(4)}</span>
      )}
    </div>
  );
}
