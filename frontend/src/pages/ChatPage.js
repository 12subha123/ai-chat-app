import { useState } from "react";

import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import ChatBox from "../components/ChatBox";

import "../styles/chat.css";

function ChatPage() {
  const [selectedConversationId, setSelectedConversationId] =
    useState(null);

  return (
    <div>
      <Sidebar
        onConversationSelect={setSelectedConversationId}
      />

      <main className="chat-container">
        <div className="chat-header">
          <Header />
        </div>

        <div className="chat-content">
          <ChatBox
            conversationId={selectedConversationId}
          />
        </div>
      </main>
    </div>
  );
}

export default ChatPage;