const API_BASE_URL = "http://127.0.0.1:8001";

export async function sendMessage(
  message,
  conversationId = null,
  attachmentPath = null
) {
  const response = await fetch(
    `${API_BASE_URL}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        conversation_id: conversationId,
        attachment_path: attachmentPath,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Backend returned status ${response.status}`
    );
  }

  return response.json();
}

export async function createConversation(title) {
  const response = await fetch(
    `${API_BASE_URL}/conversations`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        title: title,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Failed to create conversation. Status: ${response.status}`
    );
  }

  return response.json();
}

export async function getUserConversations(userId) {
  const response = await fetch(
    `${API_BASE_URL}/conversations/user/${userId}`
  );

  if (!response.ok) {
    throw new Error(
      `Failed to load conversations. Status: ${response.status}`
    );
  }

  return response.json();
}

export async function getConversationMessages(conversationId) {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}/messages`
  );

  if (!response.ok) {
    throw new Error(
      `Failed to load messages. Status: ${response.status}`
    );
  }

  return response.json();
}