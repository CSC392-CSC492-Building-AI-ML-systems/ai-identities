"use client";

import Pill from "@/components/pill";
import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";

const sections = [
  {
    title: "AI-powered LLM identifier & knowledge hub — with expert human insight",
    subtitle: "LLMDetective identifies large language models (LLMs) across the web, offering a structured, searchable wiki of every model we find — from ChatGPT to open-source tools. Explore the technology behind today’s AI, backed by transparent data and human curation.",
    pill: null,
    bg: "",
    cta: null,
  },
  {
    title: null,
    subtitle: null,
    pill: null,
    bg: "",
    cta: (
      <div className="w-full flex flex-col items-center">
        <div className="w-2/5 mx-auto flex flex-col items-center mb-12">
          <h2 className="text-4xl md:text-5xl text-[#2D2A5A] dark:text-[#F3F3FF] text-center mb-8">Identify, Search, and Explore</h2>
        </div>
        <div className="w-full flex flex-col md:flex-row justify-center gap-8">
          <div className="flex-1 max-w-xs bg-[#2D2A5A] rounded-2xl shadow-lg p-8 flex flex-col items-start text-white">
            <div className="mb-4 text-3xl">🕵️‍♂️</div>
            <div className="text-2xl mb-2 font-semibold">Identify</div>
            <div className="text-base opacity-80">Use our tools to identify which LLM generated a given text, leveraging advanced detection and comparison algorithms.</div>
          </div>
          <div className="flex-1 max-w-xs bg-[#2D2A5A] rounded-2xl shadow-lg p-8 flex flex-col items-start text-white">
            <div className="mb-4 text-3xl">🔍</div>
            <div className="text-2xl mb-2 font-semibold">Search</div>
            <div className="text-base opacity-80">Find detailed information on a wide range of LLMs, from ChatGPT to open-source models, all in one place.</div>
          </div>
          <div className="flex-1 max-w-xs bg-[#2D2A5A] rounded-2xl shadow-lg p-8 flex flex-col items-start text-white">
            <div className="mb-4 text-3xl">🧑‍💻</div>
            <div className="text-2xl mb-2 font-semibold">Explore</div>
            <div className="text-base opacity-80">Dive into the technology and data behind today’s AI, curated and explained by experts.</div>
          </div>
        </div>
      </div>
    ),
  },
  {
    title: "Methods of Identification",
    subtitle: null,
    pill: null,
    bg: "",
    cta: null,
  },
  {
    title: "Get Started",
    subtitle: "Try our Identify tool or browse the Wiki.",
    pill: "Start Now",
    bg: "bg-blue-100",
    cta: (
      <div className="flex gap-4 mt-6">
        <Button
          className="!bg-[#2D2A5A] text-white hover:outline hover:outline-2 hover:outline-[#F3F3FF] border-none"
          onClick={() => window.location.href = "/identify"}
          style={{ backgroundColor: '#2D2A5A' }}
        >
          Identify
        </Button>
        <Button
          className="!bg-[#2D2A5A] text-white hover:outline hover:outline-2 hover:outline-[#F3F3FF] border-none"
          onClick={() => window.location.href = "/wiki"}
          style={{ backgroundColor: '#2D2A5A' }}
        >
          Wiki
        </Button>
      </div>
    ),
  },
  {
    title: null,
    subtitle: null,
    pill: null,
    bg: "bg-[#2D2A5A]",
    cta: (
      <div className="flex flex-col items-center justify-center w-full">
        <img src="/llmdetlogo.png" alt="Logo" width={32} height={32} className="mb-2" />
        <span className="text-[#F3F3FF] text-sm">&copy; {new Date().getFullYear()} LLM Detective. All rights reserved.</span>
      </div>
    ),
  },
];

function useSectionFadeIn(numSections: number) {
  const [visible, setVisible] = useState(Array(numSections).fill(false));
  const refs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    const observer = new window.IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = refs.current.findIndex((el) => el === entry.target);
            if (idx !== -1 && !visible[idx]) {
              setVisible((prev) => {
                const updated = [...prev];
                updated[idx] = true;
                return updated;
              });
            }
          }
        });
      },
      { threshold: 0.3 }
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [numSections, visible]);

  return { refs, visible };
}

export default function HomePage() {
  return (
    <main className="bg-[#050a1f] pt-24">
      {sections.map((section, i) => (
        <section
          key={i}
          className={`${i === sections.length - 1 ? 'min-h-[20vh] bg-[#2D2A5A]' : 'min-h-[60vh]'} w-full flex flex-col items-center justify-center`}
        >
          <div className="w-2/5 mx-auto flex flex-col items-center">
            {section.pill && <Pill text={section.pill} />}
            {section.title && (
              <h1 className={`text-5xl md:text-6xl mt-8 ${i === 0 ? 'mb-8' : 'mb-4'} text-[#2D2A5A] dark:text-[#F3F3FF] text-center drop-shadow-lg`}>
                {section.title}
              </h1>
            )}
            {section.subtitle && (
              <p className={`text-lg md:text-xl text-[#535C91] dark:text-[#F3F3FF] text-center mb-2 ${i === 0 ? 'mt-4' : ''}`}>
                {section.subtitle}
              </p>
            )}
          </div>
          {section.cta}
        </section>
      ))}
    </main>
  );
}
