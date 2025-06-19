"use client";
import React, { useState } from "react";

const MOCK_LLM_APPS = [
  {
    id: 1,
    name: "gpt-4o",
    type: "Model",
    releaseDate: "13/5/2024",
    developer: "OpenAI",
    background:
      "Multiple versions of GPT-4o were originally secretly launched under different names on Large Model Systems Organization's (LMSYS) Chatbot Arena as three different models. These three models were called gpt2-chatbot, im-a-good-gpt...",
    color: "bg-red-400",
  },
  {
    id: 2,
    name: "Grammarly AI",
    type: "App",
    releaseDate: "25/04/2023",
    developer: "Grammarly",
    background:
      "Grammarly is an English language writing assistant software tool. It reviews the spelling, grammar, and tone of a piece of writing as well as identifying possible instances of plagiarism.",
    color: "bg-sky-400",
  },
  {
    id: 3,
    name: "Visionary Bot",
    type: "Tool",
    releaseDate: "10/01/2024",
    developer: "VisionAI Labs",
    background:
      "Visionary Bot is a multimodal AI assistant capable of understanding images and text, providing creative suggestions, and automating visual workflows.",
    color: "bg-purple-500",
  },
];

export default function WikiPage() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  // Filtered data (mock, no real pagination)
  const filtered = MOCK_LLM_APPS.filter((item) =>
    item.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main className="flex flex-col items-center min-h-screen py-8">
      <h1 className="text-4xl font-normal mb-8 mt-4">Wiki</h1>
      <div className="flex w-full max-w-2xl mb-8">
        <input
          type="text"
          placeholder="> Type to search models..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 px-4 py-2 rounded-l bg-[#393E7C] text-[#F3F3FF] placeholder-[#9290C3] focus:outline-none"
        />
        <button className="px-6 py-2 rounded-r bg-cyan-400 text-white font-semibold hover:bg-cyan-500 transition">Search</button>
      </div>
      <h2 className="text-xl font-semibold mb-4 w-full max-w-2xl">Catalogued LLMs & Apps</h2>
      <div className="flex flex-col gap-6 w-full max-w-2xl">
        {filtered.map((item) => (
          <div
            key={item.id}
            className={`group rounded-xl shadow-md p-6 text-white flex flex-col gap-2 ${item.color}`}
          >
            <div className="flex justify-between items-center mb-2">
              <span className="text-2xl font-bold">{item.name}</span>
              <span className="bg-gray-700/60 text-xs px-3 py-1 rounded-full font-semibold">{item.type}</span>
            </div>
            <div className="text-sm mb-1">
              <span className="font-semibold">Release Date:</span> {item.releaseDate}
            </div>
            <div className="text-sm mb-1">
              <span className="font-semibold">Developer(s):</span> {item.developer}
            </div>
            <div className="text-sm opacity-0 max-h-0 group-hover:opacity-100 group-hover:max-h-40 group-hover:mt-2 transition-all duration-300 overflow-hidden">
              <span className="font-semibold">Background:</span> {item.background}
            </div>
          </div>
        ))}
      </div>
      {/* Pagination Controls (mock) */}
      <div className="flex gap-2 mt-8 items-center">
        <button className="px-3 py-1 rounded bg-gray-200 text-gray-500 font-semibold" disabled>
          &#8592;
        </button>
        <button className="px-3 py-1 rounded bg-gray-300 text-gray-700 font-semibold" disabled>
          1
        </button>
        <button className="px-3 py-1 rounded bg-gray-200 text-gray-500 font-semibold" disabled>
          2
        </button>
        <span className="px-3 py-1 text-gray-400">...</span>
        <button className="px-3 py-1 rounded bg-gray-200 text-gray-500 font-semibold" disabled>
          7
        </button>
        <button className="px-3 py-1 rounded bg-gray-200 text-gray-500 font-semibold" disabled>
          &#8594;
        </button>
      </div>
    </main>
  );
}
