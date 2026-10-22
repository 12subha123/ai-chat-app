from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: int | None = None
    attachment_path: str | None = None

class ChatResponse(BaseModel):
    message: str
    image_path: str | None = None
    video_path: str | None = None
