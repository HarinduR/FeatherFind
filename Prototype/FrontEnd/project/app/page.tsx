"use client";

import { useState, useEffect, useCallback } from "react";
import { Bird } from "lucide-react";
import { MessageList, type Message } from "@/components/chat/message-list";
import { ChatInput } from "@/components/chat/chat-input";
import { AuthButtons } from "@/components/auth/auth-buttons";
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
  const [isIntroAnimationComplete, setIsIntroAnimationComplete] = useState(false);

  useEffect(() => {
    setIsIntroAnimationComplete(false);
    const timer = setTimeout(() => {
      setIsIntroAnimationComplete(true);
    }, 800);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    fetch("/api/auth/connect")
      .then((res) => res.json())
      .then((data) => {
        if (data.success) console.log("✅ MongoDB connected!");
        else console.error("❌ MongoDB connection failed:", data.error);
      })
      .catch((error) => console.error("❌ API request failed:", error));
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const ripple = document.createElement('div');
    ripple.className = 'water-ripple';
    ripple.style.left = `${e.clientX}px`;
    ripple.style.top = `${e.clientY}px`;
    document.querySelector('.dynamic-background')?.appendChild(ripple);
    
    setTimeout(() => {
      ripple.remove();
    }, 2000);
  }, []);

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
      formData.append('image', file);

      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data && data.result) {
        setMessages((prev) => [
          ...prev,
          {
            id: nanoid(),
            role: "assistant",
            content: `${data.result}. ${data.confidence || ''}`,
            type: "text",
          },
        ]);
      } else {
        throw new Error("Could not process the image.");
      }
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
    <>
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <filter id="water">
          <feTurbulence type="fractalNoise" baseFrequency="0.01 0.005" numOctaves="3" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="5" />
        </filter>
      </svg>
      <div className="dynamic-background" onMouseMove={handleMouseMove}>
        <div className="water-effect" />
      </div>
      <div className="flex flex-col h-screen bg-transparent content-wrapper">
        <header className="fixed top-0 left-0 right-0 border-b bg-card/80 backdrop-blur-sm z-10">
          <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bird className={`h-6 w-6 text-green-500 ${!isIntroAnimationComplete ? 'logo-intro' : ''}`} />
              <h1 className={`text-xl font-semibold tracking-tight ${!isIntroAnimationComplete ? 'title-intro' : ''}`}>
                FeatherFind
              </h1>
            </div>
            <AuthButtons />
          </div>
        </header>
        
        <main className="flex-1 flex flex-col pt-[57px] pb-[76px]">
          <MessageList messages={messages} isLoading={isLoading} />
        </main>

        <div className="fixed bottom-0 left-0 right-0 bg-background/80 backdrop-blur-sm border-t">
          <ChatInput
            onSubmit={handleSubmit}
            onImageUpload={handleImageUpload}
            isLoading={isLoading}
          />
        </div>
      </div>
    </>
  );
}