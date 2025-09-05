"use client";

import React, { useRef, useState, useEffect } from "react";

// Add custom CSS for fade-in animations
const fadeInStyles = `
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .fade-in-up {
    opacity: 0;
    animation: fadeInUp 0.6s ease-out forwards;
  }
  
  .fade-in-delay-1 {
    opacity: 0;
    animation: fadeInUp 0.6s ease-out 0.1s forwards;
  }
  
  .fade-in-delay-2 {
    opacity: 0;
    animation: fadeInUp 0.6s ease-out 0.2s forwards;
  }
  
  .fade-in-delay-3 {
    opacity: 0;
    animation: fadeInUp 0.6s ease-out 0.3s forwards;
  }
  
  .fade-start {
    opacity: 0;
  }
  
  .fade-start.animate {
    animation: fadeInUp 0.6s ease-out forwards;
  }
  
  .fade-start.animate.delay-1 {
    animation: fadeInUp 0.6s ease-out 0.1s forwards;
  }
  
  .fade-start.animate.delay-2 {
    animation: fadeInUp 0.6s ease-out 0.2s forwards;
  }
  
  .fade-start.animate.delay-3 {
    animation: fadeInUp 0.6s ease-out 0.3s forwards;
  }
  
  @keyframes fadeOutDown {
    from {
      opacity: 1;
      transform: translateY(0);
    }
    to {
      opacity: 0;
      transform: translateY(30px);
    }
  }
  
  @keyframes fadeInUpButton {
    from {
      opacity: 0;
      transform: translateY(-30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .button-fade-out {
    animation: fadeOutDown 0.3s ease-in forwards;
  }
  
  .button-fade-in {
    animation: fadeInUpButton 0.3s ease-out forwards;
  }
`;

export default function IdentifyPage() {
  // Inject styles on component mount
  useEffect(() => {
    const styleId = "fade-in-styles";
    if (!document.getElementById(styleId)) {
      const styleElement = document.createElement("style");
      styleElement.id = styleId;
      styleElement.textContent = fadeInStyles;
      document.head.appendChild(styleElement);
    }
  }, []);

  // reference to the uploaded file
  const fileInputRef = useRef<HTMLInputElement>(null);
  // used to display "uploading" messages, and to prevent file upload
  // while another file is being uploaded
  const [uploading, setUploading] = useState(false);
  // used to display upload status messages
  const [fullResults, setFullResults] = useState<{ [key: string]: number }>({});
  const [uploadStatus, setUploadStatus] = useState<{
    type: "success" | "error" | null;
    message: string;
  }>({ type: null, message: "" });
  // store analysis results, persisted in localStorage
  const [results, setResults] = useState<{
    [key: string]: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("identify_results");
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {}
      }
    }
    return null;
  });

  // Animation trigger key to force re-render with animations
  const [animationKey, setAnimationKey] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  // Trigger animations when showing identify page
  useEffect(() => {
    if (!uploading && !results) {
      setAnimationKey((prev) => prev + 1);
      // Small delay to ensure DOM is ready, then start animations
      const timer = setTimeout(() => {
        setIsVisible(true);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setIsVisible(false);
    }
  }, [uploading, results]);

  // Load fullResults from localStorage on component mount
  useEffect(() => {
    if (typeof window !== "undefined" && results) {
      const savedFullResults = localStorage.getItem("identify_full_results");
      if (savedFullResults) {
        try {
          setFullResults(JSON.parse(savedFullResults));
        } catch {}
      }
    }
  }, [results]);

  // store the uploaded file before analysis
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  // Button transition state
  const [buttonTransition, setButtonTransition] = useState<"idle" | "fadeOut" | "fadeIn">("idle");

  // event handler for file selection (does not upload immediately)
  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] || null;
    if (file && !uploadedFile) {
      // Transitioning from upload to identify
      setButtonTransition("fadeOut");
      setTimeout(() => {
        setUploadedFile(file);
        setUploadStatus({ type: null, message: "" });
        setResults(null);
        setButtonTransition("fadeIn");
        setTimeout(() => setButtonTransition("idle"), 200);
      }, 200);
    } else if (file) {
      setUploadedFile(file);
      setUploadStatus({ type: null, message: "" });
      setResults(null);
    }
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
        setFullResults(result.raw_classifier_result.all_scores);
        if (typeof window !== "undefined") {
          localStorage.setItem("identify_results", JSON.stringify(result.analysis));
          localStorage.setItem("identify_full_results", JSON.stringify(result.raw_classifier_result.all_scores));
        }
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

  // Handler for deleting uploaded file
  function handleDeleteFile() {
    // Transitioning from identify to upload
    setButtonTransition("fadeOut");
    setTimeout(() => {
      setUploadedFile(null);
      setUploadStatus({ type: null, message: "" });
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setButtonTransition("fadeIn");
      setTimeout(() => setButtonTransition("idle"), 200);
    }, 200);
  }

  // Handler for button click - either file select or identify
  function handleButtonClick() {
    if (!uploadedFile) {
      fileInputRef.current?.click();
    } else {
      handleIdentify();
    }
  }

  return (
    <main className="min-h-screen bg-[#050a1f] flex flex-col items-center">
      {/* Show header ONLY if not uploading and no results */}
      {!uploading && !results && (
        <div key={animationKey} className="mt-32 w-full max-w-4xl bg-transparent rounded-2xl p-10">
          <h1 className="text-4xl font-bold text-[#F3F3FF] mb-8 text-center fade-in-up">Identify the LLM</h1>
          <p className="text-base text-[#F3F3FF] mb-10 text-center max-w-3xl mx-auto leading-relaxed fade-in-delay-1">
            Prompt your LLM 14 times with the following text:
          </p>
          <p className="text-base text-[#F3F3FF] mb-10 text-center max-w-3xl mx-auto leading-relaxed fade-in-delay-1">
            "Invent a new taste unknown to humans. Describe (in less than 800 words) how it feels, what foods or
            cuisines feature it, and how it could transform food culture and impact health. Speak like a professor and
            only use vocabularies, wordings, etc professors use.".
          </p>
          <p className="text-base text-[#F3F3FF] mb-10 text-center max-w-3xl mx-auto leading-relaxed fade-in-delay-1">
            Format the responses as a JSON array like in this example:
          </p>
          <div className="bg-[#1a1a2e] rounded-lg p-6 mb-6 border border-[#3D3A6A] max-w-3xl mx-auto">
            <pre className="text-[#26FF9A] text-sm overflow-x-auto whitespace-pre-wrap">
              {`[
  "The ontological implications of a novel gastronomic experience are a subject of great fascination and inquiry. I propose to introduce a...",
  "The ontological implications of a novel gastronomical sensation are a subject of great fascination in the realm of culinary science. I propose to introduce a new taste, which I shall term...",
  "..."
]`}
            </pre>
          </div>
          <div className="flex flex-col items-center fade-in-delay-2">
            <button
              onClick={handleButtonClick}
              className={`inline-block bg-[#2D2A5A] text-[#F3F3FF] px-10 py-6 rounded-xl text-xl font-semibold cursor-pointer mb-4 flex items-center gap-4 justify-center min-w-[200px] h-[88px] ${
                uploadedFile && buttonTransition === "idle"
                  ? "hover:outline hover:outline-2 hover:outline-[#F3F3FF] hover:scale-105 transition-all duration-200"
                  : !uploadedFile && buttonTransition === "idle"
                  ? "hover:scale-105 transition-all duration-200"
                  : ""
              } ${
                buttonTransition === "fadeOut"
                  ? "button-fade-out"
                  : buttonTransition === "fadeIn"
                  ? "button-fade-in"
                  : ""
              }`}
              disabled={uploading || buttonTransition !== "idle"}
              style={{
                transform: buttonTransition === "idle" ? "translateY(0)" : undefined,
              }}
            >
              {!uploadedFile ? (
                <>
                  <div className="upload-animate flex items-center justify-center hover:animate-bounce">
                    <img src="/upload1.png" alt="Upload 1" className="h-8 w-auto" />
                  </div>
                  <div className="flex items-center justify-center">
                    <img src="/upload2.png" alt="Upload 2" className="h-4 w-auto" />
                  </div>
                </>
              ) : (
                <span>Identify</span>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                className="hidden"
                onChange={handleFileSelect}
                disabled={uploading}
              />
            </button>

            {/* Show file preview if a file is selected */}
            {uploadedFile && (
              <div className="fade-in-delay-3">
                <FilePreviewToggle file={uploadedFile} onDelete={handleDeleteFile} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Show only "Identifying..." while uploading */}
      {uploading && (
        <div className="flex flex-1 items-center justify-center w-full min-h-screen pt-20">
          <div className="flex flex-col items-center gap-8 mt-[-24%]">
            {/* Main loading animation */}
            <div className="relative">
              {/* Outer rotating ring */}
              <div className="w-32 h-32 border-4 border-[#2D2A5A] rounded-full animate-spin border-t-[#26FF9A] shadow-lg"></div>

              {/* Inner pulsing circle */}
              <div className="absolute inset-4 bg-gradient-to-br from-[#2D2A5A] to-[#26FF9A] rounded-full animate-pulse opacity-60"></div>
            </div>

            {/* Animated text + dots */}
            <div className="flex flex-col items-center gap-2">
              <span className="text-lg font-semibold text-[#2D2A5A]">
                <h2 className="text-3xl font-bold text-[#F3F3FF] mb-8 text-center">Identifying</h2>
              </span>
              <div className="flex gap-1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="w-2 h-2 bg-[#26FF9A] rounded-full animate-pulse"
                    style={{
                      animationDelay: `${i * 0.2}s`,
                      animationDuration: "1s",
                    }}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Show only results after upload */}
      {results && !uploading && (
        <>
          <div className="mt-32 w-full max-w-4xl bg-[#050a1f] rounded-2xl p-10 shadow-lg fade-in-up">
            <h2 className="text-4xl font-bold text-[#F3F3FF] mb-8 text-center fade-in-delay-1">
              Identification Results
            </h2>

            {/* Results Table */}
            <div className="bg-[#050a1f] rounded-xl overflow-hidden border border-[#3D3A6A] fade-in-delay-2">
              {/* Table Header */}
              <div
                className="grid bg-[#2D2A5A] border-b border-[#3D3A6A]"
                style={{ gridTemplateColumns: "1fr 2fr 1.5fr" }}
              >
                <div className="p-4 text-[#F3F3FF] font-bold text-lg">Model Family</div>
                <div className="p-4 text-[#F3F3FF] font-bold text-lg">Model Name</div>
                <div className="p-4 text-[#F3F3FF] font-bold text-lg text-center">Cosine Similarity Score</div>
              </div>

              {/* Table Body */}
              {Object.entries(results)
                .slice(0, 3)
                .map(([name, score], index) => {
                  const [family, modelName] = name.includes("_") ? name.split("_") : ["Unknown", name];

                  console.log(modelName);
                  return (
                    <div
                      key={name}
                      className={`grid bg-[#050a1f] ${index !== 2 ? "border-b border-[#3D3A6A]" : ""}`}
                      style={{ gridTemplateColumns: "1fr 2fr 1.5fr" }}
                    >
                      <div className="p-4 text-[#F3F3FF] text-lg capitalize">{family}</div>
                      <div className="p-4 text-[#F3F3FF] text-lg capitalize">{modelName}</div>
                      {modelName !== "unknown" ? (
                        <div className="p-4">
                          <ChartBar score={score} />
                        </div>
                      ) : null}
                    </div>
                  );
                })}
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8 fade-in-delay-3">
              <button
                onClick={() => {
                  if (!fullResults) return;
                  const top6 = Object.entries(fullResults)
                    .sort((a, b) => b[1] - a[1])
                    .reduce((acc, [k, v]) => {
                      acc[k] = v;
                      return acc;
                    }, {} as { [key: string]: number });
                  const blob = new Blob([JSON.stringify(top6, null, 2)], {
                    type: "text/plain",
                  });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "advanced_results.txt";
                  document.body.appendChild(a);
                  a.click();
                  setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }, 100);
                }}
                disabled={!fullResults || Object.keys(fullResults).length === 0}
                className="px-6 py-3 bg-[#2D2A5A] text-[#F3F3FF] rounded-lg font-medium border border-[#3D3A6A] hover:outline hover:outline-2 hover:outline-[#F3F3FF] transition-transform duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Advanced Results
              </button>

              <button
                onClick={() => {
                  setResults(null);
                  setFullResults({});
                  setUploadStatus({ type: null, message: "" });
                  if (typeof window !== "undefined") {
                    localStorage.removeItem("identify_results");
                    localStorage.removeItem("identify_full_results");
                  }
                }}
                className="px-6 py-3 bg-[#2D2A5A] text-[#F3F3FF] rounded-lg font-medium border border-[#3D3A6A] hover:outline hover:outline-2 hover:outline-[#F3F3FF] transition-transform duration-300 hover:scale-105"
              >
                Reset
              </button>
            </div>
          </div>
        </>
      )}
    </main>
  );
}

// File preview component for JSON files
function FilePreview({ file }: { file: File }) {
  const [content, setContent] = useState<string | null>(null);
  useEffect(() => {
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
    return () => {
      cancelled = true;
    };
  }, [file]);
  if (!content) return <div className="text-[#aaa] italic">Loading preview...</div>;
  return <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{content}</pre>;
}

// File preview toggle component for JSON files
function FilePreviewToggle({ file, onDelete }: { file: File; onDelete: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-6 w-full max-w-xl mx-auto flex flex-col items-center">
      <div className="flex items-center gap-2 w-full max-w-md">
        <button
          className="flex-1 bg-[#2D2A5A] text-[#F3F3FF] rounded-lg p-4 shadow-inner font-semibold flex items-center justify-between hover:outline hover:outline-1 hover:outline-[#F3F3FF] transition-all duration-200"
          onClick={() => setOpen((v) => !v)}
          type="button"
        >
          <span>Preview: {file.name}</span>
          <span className="ml-2">{open ? "▲" : "▼"}</span>
        </button>
        <button
          onClick={onDelete}
          className="bg-[#2D2A5A] hover:bg-red-600 text-white p-4 rounded-lg transition-colors duration-200 flex items-center justify-center"
          title="Delete file"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M3 6h18"></path>
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
            <path d="M8 6V4c0-1 1-2 2-2h4c0-1 1-2 2-2v2"></path>
            <line x1="10" y1="11" x2="10" y2="17"></line>
            <line x1="14" y1="11" x2="14" y2="17"></line>
          </svg>
        </button>
      </div>
      {open && (
        <div
          style={{
            fontFamily: "monospace",
            fontSize: "0.95rem",
            maxHeight: 320,
            overflow: "auto",
            border: "1px solid #3D3A6A",
          }}
          className="bg-[#2D2A5A] text-[#F3F3FF] rounded-b-lg p-4 w-full max-w-md"
        >
          <FilePreview file={file} />
        </div>
      )}
    </div>
  );
}

// Animated Identifying... with cycling dots
function IdentifyingDots() {
  const [dotCount, setDotCount] = useState(1);
  useEffect(() => {
    const interval = setInterval(() => {
      setDotCount((c) => (c % 3) + 1);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-1">
      <span className="text-[#F3F3FF] text-3xl font-bold bg-gradient-to-r from-[#F3F3FF] to-[#26FF9A] bg-clip-text text-transparent">
        Identifying
      </span>
      <div className="flex gap-1 ml-1 items-end">
        {[1, 2, 3].map((dot) => (
          <span
            key={dot}
            className={`text-3xl font-bold transition-all duration-200 ${
              dotCount >= dot
                ? "text-[#26FF9A] opacity-100 transform scale-110"
                : "text-[#2D2A5A] opacity-50 transform scale-90"
            }`}
          >
            •
          </span>
        ))}
      </div>
    </div>
  );
}

// Chart bar component for displaying score with animated progress bar
function ChartBar({ score }: { score: number }) {
  // Gradient: 0% = #2E473C, 100% = #26FF9A
  function percentToGradientColor(p: number) {
    console.log(p);
    p = Math.max(0, Math.min(100, p));
    const c1 = [46, 71, 60];
    const c2 = [38, 255, 154];
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * (p / 100));
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * (p / 100));
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * (p / 100));
    return `rgb(${r},${g},${b})`;
  }

  const [displayPercent, setDisplayPercent] = useState(0);

  useEffect(() => {
    const duration = 1000; // ms
    const startTime = performance.now();

    function animate(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      setDisplayPercent(Math.round(score * progress));
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    }

    setDisplayPercent(0);
    requestAnimationFrame(animate);
  }, [score]);

  const barColor = percentToGradientColor(displayPercent);

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Score display */}
      <span className="text-[#F3F3FF] text-lg font-bold">0.{displayPercent}</span>

      {/* Progress bar */}
      <div className="w-full h-3 bg-[#2D2A5A] rounded-full relative overflow-hidden border border-[#3D3A6A]">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${displayPercent}%`,
            background: barColor,
            minWidth: displayPercent > 0 ? "2px" : "0px",
          }}
        ></div>
      </div>
    </div>
  );
}
