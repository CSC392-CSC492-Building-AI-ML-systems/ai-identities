import { NextRequest, NextResponse } from "next/server";
import { getDatabase } from "@/lib/mongodb";

export async function GET() {
  try {
    const db = await getDatabase();
    const collection = db.collection("uploaded_files");

    // Get all uploaded files
    const files = await collection
      .find({})
      .sort({ uploadedAt: -1 }) // Most recent first
      .limit(20) // Limit to 20 files
      .project({
        filename: 1,
        uploadedAt: 1,
        size: 1,
        "metadata.originalName": 1,
        "metadata.mimeType": 1,
      }) // Don't return the full content for performance
      .toArray();

    return NextResponse.json({ files }, { status: 200 });
  } catch (error) {
    console.error("Error fetching files:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
