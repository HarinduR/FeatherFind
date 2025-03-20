import { NextResponse } from "next/server";
import connectDB from "@/lib/db"; 

export async function GET() {
  try {
    console.log("Attempting to connect to MongoDB...");
    await connectDB(); 
    return NextResponse.json({ success: true, message: "Connected to MongoDB!" });
  } catch (error) {
    const errMessage = error instanceof Error ? error.message : "An unknown error occurred";
    console.error("Error in API route:", errMessage);
    return NextResponse.json({ success: false, error: errMessage }, { status: 500 });
  }
}