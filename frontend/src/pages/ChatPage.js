import { useState } from "react";

import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import ChatBox from "../components/ChatBox";
import VoiceRecorder from "../components/voiceRecorder";

import "../styles/chat.css";

function ChatPage() {
  const [selectedConversationId, setSelectedConversationId] =
    useState(null);

  const [voiceText, setVoiceText] = useState("");

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
            voiceText={voiceText}
          />

          <VoiceRecorder
            onTranscription={(text) => {
              setVoiceText(text);
            }}
          />
        </div>
      </main>
    </div>
  );
}

export default ChatPage;