"use client";

import React, { useRef } from "react";

export default function IdentifyPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <main className="min-h-screen bg-[#050a1f] pt-24 flex flex-col items-center">
      <h1 className="text-3xl font-semibold text-[#2D2A5A] dark:text-[#F3F3FF] mt-80 mb-2">Identify the LLM</h1>
      <p className="text-base text-[#535C91] dark:text-[#F3F3FF] mb-10">Upload a JSON file to identify!</p>
      <label htmlFor="json-upload" className="group inline-block bg-[#2D2A5A] text-[#F3F3FF] px-10 py-6 rounded-xl text-xl font-semibold cursor-pointer transition-transform duration-300 hover:scale-105 mb-4 mt-4 flex items-center gap-4 justify-center">
        <div className="upload-animate flex items-center justify-center group-hover:animate-bounce">
          <img src="/upload1.png" alt="Upload 1" className="h-8 w-auto" />
        </div>
        <div className="flex items-center justify-center">
          <img src="/upload2.png" alt="Upload 2" className="h-4 w-auto" />
        </div>
        <input
          id="json-upload"
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
        />
      </label>
    </main>
  );
} 