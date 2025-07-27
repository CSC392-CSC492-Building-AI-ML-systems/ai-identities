"use client";

import React, { useRef, useState } from "react";

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

  //  event handler for when a file is uploaded
  async function handleFileUpload() {
    // get the uploaded file
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    // turns on "uploading" status messages
    setUploading(true);
    setUploadStatus({ type: null, message: "" });

    try {
      // used to send file as part of a multipart/form-data
      const formData = new FormData();
      formData.append("file", file);

      // send file to endpoint
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          type: "success",
          message: `File "${result.filename}" uploaded successfully!`,
        });
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
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  // render UI elements with tailwind
  return (
    <main className="min-h-screen bg-[#050a1f] pt-24 flex flex-col items-center">
      <h1 className="text-3xl font-semibold text-[#F3F3FF] dark:text-[#F3F3FF] mt-80 mb-2">Identify the LLM</h1>

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

        {/*if the file is uploading, add an uploading message*/}
        {uploading ? <span className="ml-2">Uploading...</span> : <span className="ml-2">Choose JSON File</span>}
        <input
          id="json-upload"
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={handleFileUpload}
          disabled={uploading}
        />
      </label>

      {/* Status message */}
      {uploadStatus.type && (
        <div
          className={`mt-4 p-4 rounded-lg max-w-md text-center ${
            uploadStatus.type === "success"
              ? "bg-green-100 text-green-800 border border-green-300"
              : "bg-red-100 text-red-800 border border-red-300"
          }`}
        >
          {uploadStatus.message}
        </div>
      )}

      {/* Loading indicator */}
      {uploading && (
        <div className="mt-4 flex items-center gap-2 text-[#F3F3FF]">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
          <span>Saving to our database...</span>
        </div>
      )}
    </main>
  );
}
