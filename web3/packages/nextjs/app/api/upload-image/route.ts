import { NextRequest, NextResponse } from "next/server";
import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";

export const config = {
  api: {
    bodyParser: false,
  },
};

export async function POST(req: NextRequest) {
  try {
    // Ensure upload directory exists
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    await mkdir(uploadDir, { recursive: true });

    // Get form data
    const formData = await req.formData();
    const analysisData = JSON.parse(formData.get("analysisData") as string);
    const file = formData.get("image") as File;

    if (!file) {
      return NextResponse.json({ error: "No image uploaded" }, { status: 400 });
    }

    // Rename the uploaded file using the provided id and the file extension
    const extension = file.name.split(".").pop();
    const filename = `${analysisData.id}.${extension}`;
    const filePath = path.join(uploadDir, filename);
    const relativePath = path.join("uploads", filename);

    // Convert file to buffer and write
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(filePath, buffer);

    // Save analysisData to the file web3/packages/nextjs/.next/analysis_data.json
    const analysisDataFile = path.join(
      process.cwd(),
      ".next",
      "analysis_data.json"
    );
    const existingData = await readFile(analysisDataFile, "utf8").catch(
      () => "[]"
    );
    const analysisDataArray = JSON.parse(existingData);
    analysisDataArray.push({ ...analysisData, imagePath: filePath });
    await writeFile(
      analysisDataFile,
      JSON.stringify(analysisDataArray, null, 2)
    );

    return NextResponse.json({
      message: "Image uploaded successfully",
      filename: `/uploads/${filename}`,
      fullPath: filePath,
      relativePath: relativePath,
    });
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Server error during upload" },
      { status: 500 }
    );
  }
}
