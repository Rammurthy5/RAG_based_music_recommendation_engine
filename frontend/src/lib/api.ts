import { RecommendRequest, RecommendResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

export async function fetchRecommendations(
  request: RecommendRequest,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  const res = await fetch(`${API_URL}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      (body as { error?: string }).error || `Request failed (${res.status})`,
    );
  }

  return res.json() as Promise<RecommendResponse>;
}
