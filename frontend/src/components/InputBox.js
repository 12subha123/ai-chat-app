import { useEffect, useState } from "react";
import { sendMessage } from "../services/api";
import "../styles/chatbox.css";

function InputBox({
  onMessageSent,
  onThinking,
  conversationId,
  voiceText,
}) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (voiceText) {
      setMessage(voiceText);
    }
  }, [voiceText]);

  async function handleSend() {
    if (
      message.trim() === "" ||
      loading ||
      conversationId === null
    ) {
      return;
    }

    try {
      setLoading(true);
      onThinking();

      const currentMessage = message;

      const data = await sendMessage(
        currentMessage,
        conversationId
      );

      await new Promise((resolve) => {
        setTimeout(resolve, 500);
      });

      console.log("Backend response:", data);

      onMessageSent(
        currentMessage,
        data
      );

      setMessage("");
    } catch (error) {
      console.error(
        "Error sending message:",
        error
      );

      alert(
        "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-input-area">
      <input
        type="text"
        placeholder="Type your message..."
        value={message}
        onChange={(event) =>
          setMessage(event.target.value)
        }
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            handleSend();
          }
        }}
        disabled={loading}
      />

      <button
        onClick={handleSend}
        disabled={loading}
      >
        {loading ? "Sending..." : "Send"}
      </button>
    </div>
  );
}

export default InputBox;