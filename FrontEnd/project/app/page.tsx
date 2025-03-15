"use client";

import { useState } from "react";
import { Bird } from "lucide-react";
import { MessageList, type Message } from "@/components/chat/message-list";
import { ChatInput } from "@/components/chat/chat-input";
import { nanoid } from "nanoid";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm FeatherFind, your bird identification assistant. You can ask me questions about birds or upload an image for identification.",
      type: "text",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (content: string) => {
    const userMessage: Message = {
      id: nanoid(),
      role: "user",
      content,
      type: "text",
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: content }),
      });

      const data = await response.json();
      
      if (data && data[0]?.text) {
        setMessages((prev) => [
          ...prev,
          {
            id: nanoid(),
            role: "assistant",
            content: data[0].text,
            type: "text",
          },
        ]);
      } else {
        throw new Error("No response from Rasa.");
      }
    } catch (error) {
      console.error("Error communicating with Rasa:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: nanoid(),
          role: "assistant",
          content: "I apologize, but I'm having trouble connecting to the server at the moment. Please try again later.",
          type: "text",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = async (file: File) => {
    const userMessage: Message = {
        id: nanoid(),
        role: "user",
        content: "I've uploaded an image for bird identification.",
        type: "image",
        imageUrl: URL.createObjectURL(file),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file); // Ensure correct key name matches Flask backend

        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData,
            headers: {
                "Accept": "application/json",
            },
            mode: 'cors',  // Ensure CORS mode is enabled
        });

        if (!response.ok) throw new Error("Failed to fetch");

        const data = await response.json();
        
        setMessages((prev) => [
            ...prev,
            {
                id: nanoid(),
                role: "assistant",
                content: `I believe this is a ${data.result}. Confidence: ${data.confidence.toFixed(2)}`,
                type: "text",
            },
        ]);
    } catch (error) {
        console.error("Error processing image:", error);
        setMessages((prev) => [
            ...prev,
            {
                id: nanoid(),
                role: "assistant",
                content: "I apologize, but I couldn't process the image at this time. Please try again later.",
                type: "text",
            },
        ]);
    } finally {
        setIsLoading(false);
    }
};

  

  return (
    <div className="flex flex-col h-screen bg-background">
      <header className="border-b bg-card">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center gap-2">
          <Bird className="h-6 w-6 text-green-500" />
          <h1 className="text-xl font-semibold tracking-tight">FeatherFind</h1>
        </div>
      </header>
      
      <main className="flex-1 flex flex-col">
        <MessageList messages={messages} isLoading={isLoading} />
        <ChatInput
          onSubmit={handleSubmit}
          onImageUpload={handleImageUpload}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}