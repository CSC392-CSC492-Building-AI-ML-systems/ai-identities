import { NextRequest, NextResponse } from "next/server";
// import { getDatabase } from "@/lib/mongodb"; // <-- Commented out

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
      return NextResponse.json(
        { error: "File must be a JSON file" },
        { status: 400 }
      );
    }

    // Read file content
    const fileContent = await file.text();
    let jsonData;

    try {
      jsonData = JSON.parse(fileContent);
    } catch (parseError) {
      return NextResponse.json(
        { error: "Invalid JSON format" },
        { status: 400 }
      );
    }

    // Extract responses from the JSON data
    let responses: string[] = [];

    if (Array.isArray(jsonData)) {
      // If it's an array, assume each item is a response
      responses = jsonData.map((item) =>
        typeof item === "string" ? item : JSON.stringify(item)
      );
    } else if (jsonData.responses && Array.isArray(jsonData.responses)) {
      // If it has a responses field
      responses = jsonData.responses.map((item: any) =>
        typeof item === "string" ? item : JSON.stringify(item)
      );
    } else if (jsonData.messages && Array.isArray(jsonData.messages)) {
      // If it has a messages field (common in chat formats)
      responses = jsonData.messages
        .filter((msg: any) => msg.role === "assistant" || msg.role === "model")
        .map((msg: any) => msg.content || JSON.stringify(msg));
    } else {
      // Fallback: try to extract any text content
      responses = [JSON.stringify(jsonData)];
    }

    if (responses.length === 0) {
      return NextResponse.json(
        { error: "No valid responses found in the JSON file" },
        { status: 400 }
      );
    }

    // Call the new FastAPI classifier service
    const classifierUrl =
      process.env.CLASSIFIER_SERVICE_URL || "http://localhost:8000";
    const response = await fetch(`${classifierUrl}/identify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(responses),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Classifier service error:", errorText);
      return NextResponse.json(
        {
          error: "Classifier service error",
          details: errorText,
        },
        { status: 500 }
      );
    }

    const classifierResult = await response.json();
    console.log(classifierResult);

    // Transform the classifier result to match the expected frontend format
    let analysis: { [key: string]: number } = {};

    if (classifierResult.prediction && classifierResult.prediction[0] !== "unknown") {
      classifierResult.prediction.forEach(([model, score]: [string, number]) => {
        const percentage = score * 100;
        // Keep model name exactly as returned (with _ and - intact)
        analysis[model] = percentage;
      });
    } else {
      // If unknown, return a default response
      analysis = {
        unknown: 100,
      };
    }

    console.log(analysis);

    return NextResponse.json(
      {
        message: "File uploaded successfully",
        filename: file.name,
        analysis,
        raw_classifier_result: classifierResult, // Include raw result for debugging
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}