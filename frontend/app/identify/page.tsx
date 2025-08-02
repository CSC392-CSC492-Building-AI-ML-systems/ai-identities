"use client";

import React, { useRef, useState } from "react";
import { Button } from "@/components/ui/button";

export default function IdentifyPage() {
  // reference to the uploaded file
  const fileInputRef = useRef<HTMLInputElement>(null);
  // used to display "uploading" messages, and to prevent file upload
  // while another file is being uploaded
  const [uploading, setUploading] = useState(false);
  // used to display upload status messages
  const [uploadStatus, setUploadStatus] = useState<{
    type: "success" | "error" | null;
    message: string;
  }>({ type: null, message: "" });
  // store analysis results
  const [results, setResults] = useState<{
    [key: string]: number;
  } | null>(null);
  // store the uploaded file before analysis
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // event handler for file selection (does not upload immediately)
  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] || null;
    setUploadedFile(file);
    setUploadStatus({ type: null, message: "" });
    setResults(null);
  }

  // event handler for when the user presses Identify
  async function handleIdentify() {
    if (!uploadedFile) return;
    setUploading(true);
    setUploadStatus({ type: null, message: "" });
    setResults(null);
    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        setUploadStatus({
          type: "success",
          message: `File \"${result.filename}\" analyzed successfully!`,
        });
        setResults(result.analysis);
      } else {
        setUploadStatus({
          type: "error",
          message: result.error || "Upload failed",
        });
      }
    } catch (error) {
      setUploadStatus({
        type: "error",
        message: "Network error occurred during upload",
      });
    } finally {
      setUploading(false);
      setUploadedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <main className="min-h-screen bg-[#050a1f] pt-24 flex flex-col items-center">
      {/* Show header ONLY if not uploading and no results */}
      {!uploading && !results && (
        <>
          <h1 className="text-3xl font-semibold text-[#F3F3FF] dark:text-[#F3F3FF] mt-80 mb-2">
            Identify the LLM
          </h1>
          <p className="text-base text-[#F3F3FF] dark:text-[#F3F3FF] mb-10">Upload a JSON file to identify!</p>
          <label
            htmlFor="json-upload"
            className="group inline-block bg-[#2D2A5A] text-[#F3F3FF] px-10 py-6 rounded-xl text-xl font-semibold cursor-pointer transition-transform duration-300 hover:scale-105 mb-4 mt-4 flex items-center gap-4 justify-center"
          >
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
              onChange={handleFileSelect}
              disabled={uploading}
            />
          </label>
          {/* Show Identify button if a file is selected */}
          {uploadedFile && (
            <>
              <Button
                className="mt-6 !bg-[#2D2A5A] text-white hover:outline hover:outline-2 hover:outline-[#F3F3FF] border-none transition-transform duration-300 hover:scale-105 animate-fade-in-up"
                style={{ backgroundColor: '#2D2A5A', alignSelf: 'center', display: 'block' }}
                onClick={handleIdentify}
                disabled={uploading}
              >
                Identify
              </Button>
              <FilePreviewToggle file={uploadedFile} />
            </>
          )}
        </>
      )}

      {/* Show only "Identifying..." while uploading */}
      {uploading && (
        <div className="flex flex-1 items-center justify-center w-full" style={{ minHeight: "60vh" }}>
          <IdentifyingDots />
        </div>
      )}

      {/* Show only results after upload */}
      {results && !uploading && (
        <div className="mt-32 w-full max-w-2xl bg-[#2D2A5A] rounded-2xl p-10 shadow-lg">
          <h2 className="text-4xl font-bold text-[#F3F3FF] mb-8 text-center">Identification Results</h2>
          <div className="space-y-8">
            {Object.entries(results).map(([name, percent]) => (
              <AnimatedBar key={name} name={name} percent={percent} />
            ))}
          </div>
          <button
            className="mt-12 w-full bg-[#23204a] text-[#F3F3FF] px-8 py-4 rounded-xl text-2xl font-bold hover:bg-[#1a1833] transition"
            onClick={() => {
              setResults(null);
              setUploadStatus({ type: null, message: "" });
            }}
          >
            New Identification
          </button>
        </div>
      )}
    </main>
  );
}

// File preview component for JSON files
function FilePreview({ file }: { file: File }) {
  const [content, setContent] = React.useState<string | null>(null);
  React.useEffect(() => {
    let cancelled = false;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (!cancelled) {
        try {
          // Pretty print JSON if possible
          const text = e.target?.result as string;
          const json = JSON.parse(text);
          setContent(JSON.stringify(json, null, 2));
        } catch {
          setContent(e.target?.result as string);
        }
      }
    };
    reader.readAsText(file);
    return () => { cancelled = true; };
  }, [file]);
  if (!content) return <div className="text-[#aaa] italic">Loading preview...</div>;
  return <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{content}</pre>;
}

// File preview toggle component for JSON files
function FilePreviewToggle({ file }: { file: File }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="mt-6 w-full max-w-xl mx-auto animate-fade-in-up">
      <button
        className="w-full bg-[#18163a] text-[#F3F3FF] rounded-lg p-4 shadow-inner font-semibold flex items-center justify-between hover:bg-[#23204a] transition mb-1"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span>Preview: {file.name}</span>
        <span className="ml-2">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ fontFamily: 'monospace', fontSize: '0.95rem', maxHeight: 320, overflow: 'auto', border: '1px solid #23204a' }} className="bg-[#18163a] text-[#F3F3FF] rounded-b-lg p-4">
          <FilePreview file={file} />
        </div>
      )}
    </div>
  );
}

// Animated Identifying... with cycling dots
function IdentifyingDots() {
  const [dotCount, setDotCount] = React.useState(1);
  React.useEffect(() => {
    const interval = setInterval(() => {
      setDotCount((c) => (c % 3) + 1);
    }, 400);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="text-[#F3F3FF] text-3xl font-semibold">
      Identifying{'.'.repeat(dotCount)}
    </div>
  );
}

// Animated bar and percent as a separate component to avoid breaking the Rules of Hooks
function AnimatedBar({ name, percent }: { name: string; percent: number }) {
  // Gradient: 0% = #2E473C, 100% = #26FF9A
  function percentToGradientColor(p: number) {
    p = Math.max(0, Math.min(100, p));
    const c1 = [46, 71, 60];
    const c2 = [38, 255, 154];
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * (p / 100));
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * (p / 100));
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * (p / 100));
    return `rgb(${r},${g},${b})`;
  }
  const [displayPercent, setDisplayPercent] = React.useState(0);
  React.useEffect(() => {
    const duration = 1000; // ms
    const startTime = performance.now();
    function animate(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      setDisplayPercent(Math.round(percent * progress));
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    }
    setDisplayPercent(0);
    requestAnimationFrame(animate);
  }, [percent]);
  const barColor = percentToGradientColor(displayPercent);
  return (
    <div className="flex items-center gap-8">
      <div className="w-40 text-[#F3F3FF] text-2xl font-bold capitalize">{name}</div>
      <div className="flex-1 h-12 flex items-center min-w-[300px]">
        <div className="w-full h-full bg-[#050a1f] rounded-lg relative overflow-hidden border-4 border-[#050a1f]">
          <div
            className="h-full rounded-lg transition-all duration-1000 ease-out absolute top-0 left-0"
            style={{ width: `${displayPercent}%`, background: barColor }}
          ></div>
        </div>
      </div>
      <div className="w-24 text-[#F3F3FF] text-2xl font-bold text-right">{displayPercent}%</div>
    </div>
  );
}

/* Add animation to globals.css or tailwind config:
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(32px); }
  100% { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up {
  animation: fade-in-up 0.5s cubic-bezier(0.4,0,0.2,1) both;
}
*/
