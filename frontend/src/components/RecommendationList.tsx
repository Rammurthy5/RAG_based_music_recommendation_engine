import { Recommendation } from "@/lib/types";
import { RecommendationCard } from "./RecommendationCard";

interface RecommendationListProps {
  recommendations: Recommendation[];
}

export function RecommendationList({ recommendations }: RecommendationListProps) {
  if (recommendations.length === 0) {
    return (
      <p className="text-gray-500 text-center text-sm mt-8">
        No recommendations found. Try a different description.
      </p>
    );
  }

  return (
    <div className="space-y-3 mb-6">
      {recommendations.map((rec, i) => (
        <RecommendationCard key={`${rec.track_id || i}`} recommendation={rec} />
      ))}
    </div>
  );
}
