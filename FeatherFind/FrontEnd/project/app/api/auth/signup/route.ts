import { NextResponse } from "next/server";
import connectDB from "@/lib/db";
import mongoose from "mongoose";
import bcrypt from "bcryptjs";

// Set up automatic body parsing for Next.js API routes
export const config = {
  api: {
    bodyParser: true, // Ensure the body is parsed automatically as JSON
  },
};

// Define User Schema for MongoDB
const UserSchema = new mongoose.Schema({
  username: { type: String, required: true },
  password: { type: String, required: true },
});

const User = mongoose.models.User || mongoose.model("User", UserSchema);

export async function POST(req: Request) {
  try {
    await connectDB(); // Ensure DB connection

    // Extract data from request body
    let body;
    try {
      body = await req.json();
    } catch (error) {
      console.error("Error parsing JSON:", error);
      return NextResponse.json(
        { success: false, error: error },
        { status: 400 }
      );
    }

    console.log("Received data:", body); // Log the request data for debugging

    const { username, password } = body;

    // Check if required fields are present
    if (!username || !password) {
      return NextResponse.json(
        { success: false, error: "Username and password are required" },
        { status: 400 }
      );
    }

    // Check if the username already exists in the database
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return NextResponse.json(
        { success: false, error: "User already exists" },
        { status: 400 }
      );
    }

    // Hash the password for security
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create and save the new user
    const newUser = new User({ username, password: hashedPassword });
    await newUser.save();

    // Return success response
    return NextResponse.json(
      { success: true, message: "User registered successfully" },
      { status: 201 }
    );
  } catch (error) {
    // Handle unexpected errors
    console.error("Signup Error:", error);
    return NextResponse.json(
      { success: false, error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
