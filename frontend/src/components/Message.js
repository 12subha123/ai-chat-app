function Message({ role, content }) {
  const isUser = role === "user";

  return (
    <div className={isUser ? "message user-message" : "message ai-message"}>
      <div className="message-role">
        {isUser ? "You" : "AI"}
      </div>

      <div className="message-content">
        {content}
      </div>
    </div>
  );
}

export default Message;