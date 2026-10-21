import { useState } from "react";
import { sendMessage } from "../services/api";

export default function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function sendChatMessage(
    message,
    conversationId = null
  ) {
    if (!message.trim()) {
      return;
    }

    const userMessage = {
      role: "user",
      content: message,
    };

    setMessages((previousMessages) => [
      ...previousMessages,
      userMessage,
    ]);

    setLoading(true);
    setError(null);

    try {
      const data = await sendMessage(
        message,
        conversationId
      );

      const assistantMessage = {
        role: "assistant",
        content: data.message,
        imagePath: data.image_path || null,
      };

      setMessages((previousMessages) => [
        ...previousMessages,
        assistantMessage,
      ]);

      return data;
    } catch (err) {
      console.error("Chat request failed:", err);

      setError(
        err.message || "Failed to get a response from the AI."
      );

      return null;
    } finally {
      setLoading(false);
    }
  }

  function clearMessages() {
    setMessages([]);
    setError(null);
  }

  return {
    messages,
    loading,
    error,
    sendChatMessage,
    clearMessages,
  };
}