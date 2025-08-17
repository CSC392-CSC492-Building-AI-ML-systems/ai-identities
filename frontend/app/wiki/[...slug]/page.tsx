'use client';

import { useParams } from 'next/navigation';

export default function WikiIframePage() {
  const params = useParams();
  const slug = params?.slug as string[] | undefined;

  if (!slug || slug.length < 1) {
    return <p className="text-white mt-20">Invalid Wiki URL</p>;
  }

  // Join the segments to reconstruct the full space + page
  const fullPath = slug.join('/');
  const wikiUrl = `https://wiki.llm.test/bin/view/${fullPath}`;

  return (
    <main className="min-h-screen bg-[#050a1f] pt-20 flex flex-col items-center">
      <div className="w-full flex-1 flex justify-center">
        <iframe
          src={wikiUrl}
          title={`Wiki: ${fullPath}`}
          className="w-full max-w-7xl h-[80vh] rounded-xl border-2 border-[#2D2A5A] bg-white"
          style={{ minHeight: 400 }}
        />
      </div>
    </main>
  );
}
