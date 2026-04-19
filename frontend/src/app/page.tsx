import { SearchBar } from "@/components/SearchBar";
import { RecommendationList } from "@/components/RecommendationList";

export default function Home() {
  return (
    <main className="max-w-3xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center mb-2">
        Music Recommendations
      </h1>
      <p className="text-gray-400 text-center mb-10">
        Describe a mood, vibe, or moment — get personalized song picks.
      </p>
      <SearchBar />
      <RecommendationList />
    </main>
  );
}
