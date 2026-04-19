interface RecommendationCardProps {
  title: string;
  artist: string;
  album: string;
  genre: string[];
  reason: string;
  artworkUrl: string;
}

export function RecommendationCard({
  title,
  artist,
  album,
  genre,
  reason,
  artworkUrl,
}: RecommendationCardProps) {
  return (
    <div className="flex gap-4 rounded-xl bg-gray-800/50 border border-gray-700/50 p-4">
      <div className="w-20 h-20 flex-shrink-0 rounded-lg bg-gray-700 overflow-hidden">
        {artworkUrl ? (
          <img
            src={artworkUrl}
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
        <h3 className="font-semibold text-gray-100 truncate">{title}</h3>
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
