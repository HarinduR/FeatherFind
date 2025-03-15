"use client";

import { useEffect, useRef, useState } from "react";
import { Bird, Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";

export type Message = {
  id: string;
  content: string;
  role: "user" | "assistant";
  type: "text" | "image";
  imageUrl?: string;
};

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [typingMessages, setTypingMessages] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    messages.forEach((message) => {
      if (message.role === "assistant" && !typingMessages[message.id]) {
        let currentText = "";
        const words = message.content.split(" ");
        let currentIndex = 0;

        const interval = setInterval(() => {
          if (currentIndex < words.length) {
            currentText += (currentIndex === 0 ? "" : " ") + words[currentIndex];
            setTypingMessages((prev) => ({
              ...prev,
              [message.id]: currentText,
            }));
            currentIndex++;
          } else {
            clearInterval(interval);
          }
        }, 50);

        return () => clearInterval(interval);
      }
    });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4">
      <div className="max-w-4xl mx-auto">
        <div className="space-y-6 py-8">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-4 rounded-lg p-4 max-w-[85%] transition-all duration-200",
                message.role === "user"
                  ? "bg-muted/50 hover:bg-muted ml-auto items-end hover:shadow-[0_0_10px_rgba(34,197,94,0.2)]"
                  : "bg-accent/50 hover:bg-accent mr-auto items-start hover:shadow-[0_0_10px_rgba(34,197,94,0.2)]"
              )}
            >
              <div className={cn(
                "rounded-full bg-background p-2",
                message.role === "user" && "order-last"
              )}>
                {message.role === "user" ? (
                  <User className="h-4 w-4" />
                ) : message.type === "image" ? (
                  <Bird className="h-4 w-4 text-green-500" />
                ) : (
                  <Bot className="h-4 w-4 text-green-500" />
                )}
              </div>
              <div className={cn(
                "flex-1 space-y-2",
                message.role === "user" ? "text-right" : "text-left"
              )}>
                {message.type === "image" && message.imageUrl && (
                  <img
                    src={message.imageUrl}
                    alt="Uploaded bird"
                    className="rounded-lg max-w-sm"
                  />
                )}
                <p className="text-sm leading-relaxed">
                  {message.role === "assistant" 
                    ? (typingMessages[message.id] || "") + (
                        typingMessages[message.id]?.length === message.content.length 
                          ? "" 
                          : "▋"
                      )
                    : message.content
                  }
                </p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-pulse space-x-2">
                <span className="inline-block h-2 w-2 rounded-full bg-green-500"></span>
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 animation-delay-200"></span>
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 animation-delay-400"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}