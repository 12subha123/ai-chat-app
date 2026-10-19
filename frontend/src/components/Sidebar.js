import { useEffect, useState } from "react";

import {
  createConversation,
  getUserConversations,
} from "../services/api";

import "../styles/sidebar.css";


function Sidebar({ onConversationSelect }) {
  const [conversations, setConversations] = useState([]);
  const [selectedConversationId, setSelectedConversationId] = useState(null);

  useEffect(() => {
    async function loadConversations() {
      try {
        const data = await getUserConversations(1);

        setConversations(data);
      } catch (error) {
        console.error(
          "Failed to load conversations:",
          error
        );
      }
    }

    loadConversations();
  }, []);

  async function handleNewChat() {
    try {
      const conversation = await createConversation(
        "New Chat"
      );

      setConversations((previousConversations) => [
        conversation,
        ...previousConversations,
      ]);

      console.log(
        "New conversation created:",
        conversation
      );

      onConversationSelect(conversation.id);
    } catch (error) {
      console.error(
        "Failed to create new conversation:",
        error
      );
    }
  }

  return (
    <aside className="sidebar">
      <h2>AI Chat App</h2>

      <button onClick={handleNewChat}>
        + New Chat
      </button>

      <h3>Conversations</h3>

      {conversations.length === 0 ? (
        <p>No conversations yet.</p>
      ) : (
        <div className="conversation-list">
          {conversations.map((conversation) => (
            <div
              className={selectedConversationId === conversation.id ? "conversation-item selected" : "conversation-item"}
              key={conversation.id}
              onClick={() => {
                console.log(
                  "Selected conversation:",
                  conversation
                );
                setSelectedConversationId(conversation.id);

                onConversationSelect(conversation.id);
              }}
            >
              {conversation.title}
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

export default Sidebar;