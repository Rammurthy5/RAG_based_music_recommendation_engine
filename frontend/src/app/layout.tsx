import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Music Recommendations",
  description: "Describe a mood or vibe and get personalized song recommendations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
