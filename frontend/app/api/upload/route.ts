import { NextRequest, NextResponse } from "next/server";
import { getDatabase } from "@/lib/mongodb";

export async function POST(req: NextRequest) {
  try {
    // Parse the form data
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Check if file is JSON
    if (!file.name.endsWith(".json") && file.type !== "application/json") {
      return NextResponse.json({ error: "File must be a JSON file" }, { status: 400 });
    }

    // Read file content
    const fileContent = await file.text();
    let jsonData;

    try {
      jsonData = JSON.parse(fileContent);
    } catch (parseError) {
      return NextResponse.json({ error: "Invalid JSON format" }, { status: 400 });
    }

    // Connect to MongoDB and save the data
    const db = await getDatabase();
    const collection = db.collection("uploaded_files");

    const document = {
      filename: file.name,
      uploadedAt: new Date(),
      size: file.size,
      content: jsonData,
      metadata: {
        originalName: file.name,
        mimeType: file.type,
      },
    };

    const result = await collection.insertOne(document);

    return NextResponse.json(
      {
        message: "File uploaded successfully",
        fileId: result.insertedId,
        filename: file.name,
        uploadedAt: document.uploadedAt,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
