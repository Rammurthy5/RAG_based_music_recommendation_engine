"use client";

import { useRef, useState } from "react";
import { SearchBar } from "@/components/SearchBar";
import { RecommendationList } from "@/components/RecommendationList";
import { MetadataBar } from "@/components/MetadataBar";
import { fetchRecommendations } from "@/lib/api";
import { RecommendResponse } from "@/lib/types";

export default function Home() {
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  async function handleSearch(query: string) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await fetchRecommendations(
        { query, limit: 5 },
        controller.signal,
      );
      setResult(data);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center mb-2">
        🎵 Music Recommendations
      </h1>
      <p className="text-gray-400 text-center mb-10">
        Describe a mood, vibe, or moment — get personalized song picks.
      </p>
      <SearchBar onSearch={handleSearch} loading={loading} />

      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700/50 px-4 py-3 text-red-300 text-sm mb-6">
          {error}
        </div>
      )}

      {loading && <SkeletonList />}

      {result && (
        <>
          <RecommendationList recommendations={result.recommendations} />
          <MetadataBar metadata={result.metadata} />
        </>
      )}

      {!result && !loading && !error && (
        <p className="text-gray-500 text-center text-sm mt-12">
          Your recommendations will appear here.
        </p>
      )}
    </main>
  );
}

function SkeletonList() {
  return (
    <div className="space-y-3 mb-6">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="flex gap-4 rounded-xl bg-gray-800/50 border border-gray-700/50 p-4 animate-pulse"
        >
          <div className="w-20 h-20 flex-shrink-0 rounded-lg bg-gray-700" />
          <div className="flex-1 space-y-2 py-1">
            <div className="h-4 bg-gray-700 rounded w-2/3" />
            <div className="h-3 bg-gray-700 rounded w-1/2" />
            <div className="h-3 bg-gray-700 rounded w-5/6 mt-3" />
          </div>
        </div>
      ))}
    </div>
  );
}
