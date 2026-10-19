import {useEffect, useRef, useState } from "react";
import { getConversationMessages } from "../services/api";

import Message from "./Message";
import InputBox from "./InputBox";
import "../styles/chatbox.css";

function ChatBox({ conversationId }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Welcome to AI Chat App!",
    },
  ]);
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef(null);
 

useEffect(() => {
  messagesEndRef.current?.scrollIntoView({
    behavior: "smooth",
 
  });
}, [messages]);
useEffect(() => {
  async function loadMessages() {
    if (conversationId === null) {
      setMessages([]);
      return;
    }

    try {
      const data = await getConversationMessages(
        conversationId
      );

      setMessages(data);

      console.log(
        "Loaded conversation messages:",
        data
      );
    } catch (error) {
      console.error(
        "Failed to load conversation messages:",
        error
      );
    }
  }

  loadMessages();
}, [conversationId]);

function handleMessageSent(userMessage, aiResponse) {
  setIsThinking(false);

  setMessages((previousMessages) => [
    ...previousMessages,
    {
      role: "user",
      content: userMessage,
    },
    {
      role: "assistant",
      content: aiResponse,
    },
  ]);
}

  return (
    <div className="chat-box">
    <div className="chat-messages">
  {messages.map((message, index) => (
    <div className="chat-message" key={index}>
      <Message
        role={message.role}
        content={message.content}
      />
    </div>
  ))}

  {isThinking && (
    <div className="chat-message">
      <div className="message ai-message">
        <div className="message-role">
          AI
        </div>

        <div className="message-content thinking">
          AI is thinking...
        </div>
      </div>
    </div>
  )}

  <div ref={messagesEndRef} />
</div>

     <InputBox
  onMessageSent={handleMessageSent}
  onThinking={() => setIsThinking(true)}
  conversationId={conversationId}
/>
    </div>
  );
}

export default ChatBox;