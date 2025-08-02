import { NextRequest, NextResponse } from "next/server";
// import { getDatabase } from "@/lib/mongodb"; // <-- Commented out
import { spawn } from "child_process";
import path from "path";

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

    // --- MongoDB code commented out ---
    // const db = await getDatabase();
    // const collection = db.collection("uploaded_files");
    // const document = {
    //   filename: file.name,
    //   uploadedAt: new Date(),
    //   size: file.size,
    //   content: jsonData,
    //   metadata: {
    //     originalName: file.name,
    //     mimeType: file.type,
    //   },
    // };
    // const result = await collection.insertOne(document);

    // --- Call the Python algorithm ---
    const scriptPath = path.join(process.cwd(), "..", "algorithm.py");
    const py = spawn("python", [scriptPath]);

    // Log any errors from the Python script
    py.stderr.on("data", (data) => {
    console.error("PYTHON ERROR:", data.toString());
    });

    py.stdin.write(fileContent);
    py.stdin.end();

    let output = "";
    for await (const chunk of py.stdout) {
      output += chunk;
    }

    let analysis;
    try {
      analysis = JSON.parse(output);
    } catch {
      return NextResponse.json({ error: "Algorithm error" }, { status: 500 });
    }

    return NextResponse.json(
      {
        message: "File uploaded successfully",
        // fileId: result.insertedId, // commented out
        filename: file.name,
        // uploadedAt: document.uploadedAt, // commented out
        analysis, // <-- returned from Python
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
