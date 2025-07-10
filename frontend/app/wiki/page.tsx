// NOTE: Store credentials in .env.local as NEXT_PUBLIC_XWIKI_USERNAME and NEXT_PUBLIC_XWIKI_PASSWORD
"use client";
import React, { useEffect, useState } from "react";
import { parseStringPromise } from "xml2js";

const XWIKI_API_URL = "http://159.203.20.200:8080/xwiki/rest/wikis/xwiki/spaces/Main/pages";

// Type for a page entry from XWiki REST API
interface XWikiPage {
  id: string[];
  title: string[];
  space: string[];
  name: string[];
}

// Credentials from environment variables
const XWIKI_USERNAME = process.env.NEXT_PUBLIC_XWIKI_USERNAME || "";
const XWIKI_PASSWORD = process.env.NEXT_PUBLIC_XWIKI_PASSWORD || "";
const BASIC_AUTH = typeof window !== "undefined"
  ? "Basic " + btoa(`${XWIKI_USERNAME}:${XWIKI_PASSWORD}`)
  : "";

export default function WikiPage() {
  const [pages, setPages] = useState<XWikiPage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    async function fetchPages() {
      setLoading(true);
      setError("");
      try {
        const res = await fetch(XWIKI_API_URL, {
          headers: {
            Accept: "application/xml",
            Authorization: BASIC_AUTH,
          },
        });
        if (!res.ok) throw new Error("Network response was not ok");
        const xml = await res.text();
        const result = await parseStringPromise(xml);
        setPages((result.pages.page as XWikiPage[]) || []);
      } catch (err) {
        setError("Failed to fetch wiki pages.");
      }
      setLoading(false);
    }
    fetchPages();
  }, []);

  const filtered = pages.filter((page) =>
    page.title[0].toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main className="flex flex-col items-center min-h-screen py-8">
      <h1 className="text-4xl font-normal mb-8 mt-4">Wiki</h1>
      <div className="flex w-full max-w-2xl mb-8">
        <input
          type="text"
          placeholder="> Type to search pages..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 px-4 py-2 rounded-l bg-[#393E7C] text-[#F3F3FF] placeholder-[#9290C3] focus:outline-none"
        />
      </div>
      {loading && <div>Loading...</div>}
      {error && <div className="text-red-500">{error}</div>}
      <div className="flex flex-col gap-6 w-full max-w-2xl">
        {filtered.map((page) => (
          <div key={page.id[0]} className="rounded-xl shadow-md p-6 text-white bg-blue-500">
            <div className="flex justify-between items-center mb-2">
              <span className="text-2xl font-bold">{page.title[0]}</span>
              <a
                href={`http://159.203.20.200:8080/bin/view/${page.space[0]}/${page.name[0]}`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gray-700/60 text-xs px-3 py-1 rounded-full font-semibold"
              >
                View
              </a>
            </div>
            <div className="text-sm mb-1">
              <span className="font-semibold">ID:</span> {page.id[0]}
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
