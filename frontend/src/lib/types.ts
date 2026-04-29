export interface Recommendation {
  title: string;
  artist: string;
  album: string;
  genre: string[];
  reason: string;
  artwork_url: string;
  track_id: string;
  similarity_score: number;
}

export interface CostInfo {
  input_tokens: number;
  output_tokens: number;
  total_cost_usd: number;
}

export interface RAGConfigInfo {
  top_k: number;
  chunk_size: number;
  similarity_threshold: number;
  embedding_model: string;
}

export interface ResponseMetadata {
  source: "full_rag" | "retrieval_only" | "fallback_cache";
  prompt_id: string;
  model: string;
  rag_config: RAGConfigInfo;
  cost: CostInfo;
  latency_ms: number;
}

export interface RecommendResponse {
  query: string;
  recommendations: Recommendation[];
  metadata: ResponseMetadata;
}

export interface RecommendRequest {
  query: string;
  limit?: number;
}
