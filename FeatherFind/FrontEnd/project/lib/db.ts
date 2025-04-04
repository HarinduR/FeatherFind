import mongoose from "mongoose";

const MONGODB_URI = "mongodb+srv://thehara00:featherfind4@featherfind.6qqef.mongodb.net/featherfind?retryWrites=true&w=majority&appName=FeatherFind";

async function connectDB() {
  if (mongoose.connection.readyState >= 1) {
    console.log("Using existing database connection");
    return mongoose.connection;
  }

  try {
    console.log("Creating new database connection");
    await mongoose.connect(MONGODB_URI, {
      bufferCommands: false, // Disable command buffering
    });
    console.log("Connected to MongoDB");
    return mongoose.connection;
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    throw error; // Re-throw the error to handle it in the API route
  }
}

export default connectDB;
