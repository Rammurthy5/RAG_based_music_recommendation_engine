import { Recommendation } from "@/lib/types";

interface RecommendationCardProps {
  recommendation: Recommendation;
}

export function RecommendationCard({ recommendation }: RecommendationCardProps) {
  const { title, artist, album, genre, reason, artwork_url, similarity_score } =
    recommendation;

  return (
    <div className="flex gap-4 rounded-xl bg-gray-800/50 border border-gray-700/50 p-4 hover:bg-gray-800/80 transition-colors">
      <div className="w-20 h-20 flex-shrink-0 rounded-lg bg-gray-700 overflow-hidden">
        {artwork_url ? (
          <img
            src={artwork_url}
            alt={`${title} artwork`}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500 text-2xl">
            ♪
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-2">
          <h3 className="font-semibold text-gray-100 truncate">{title}</h3>
          {similarity_score > 0 && (
            <span className="text-xs text-gray-500 whitespace-nowrap flex-shrink-0">
              {Math.round(similarity_score * 100)}% match
            </span>
          )}
        </div>
        <p className="text-gray-400 text-sm truncate">
          {artist}
          {album && ` · ${album}`}
        </p>
        {genre.length > 0 && (
          <div className="flex gap-1.5 mt-1.5 flex-wrap">
            {genre.map((g) => (
              <span
                key={g}
                className="text-xs bg-gray-700 text-gray-300 rounded-full px-2 py-0.5"
              >
                {g}
              </span>
            ))}
          </div>
        )}
        {reason && (
          <p className="text-gray-400 text-sm mt-2 line-clamp-2">{reason}</p>
        )}
      </div>
    </div>
  );
}
