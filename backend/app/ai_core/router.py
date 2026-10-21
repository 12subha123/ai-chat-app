from enum import Enum

from .models import AIRequest


class Capability(str, Enum):
    LLM = "llm"
    KNOWLEDGE = "knowledge"
    WEB = "web"
    VOICE = "voice"
    IMAGE = "image"
    VIDEO = "video"


def route_request(request: AIRequest) -> Capability:
    message = request.message.strip().lower()

    if not message:
        return Capability.LLM

    metadata = request.metadata or {}

    # Explicit knowledge request.
    #
    # This prevents ordinary questions from being routed
    # through RAG automatically.
    if metadata.get("use_knowledge") is True:
        return Capability.KNOWLEDGE

    image_keywords = (
        "generate an image",
        "create an image",
        "make an image",
        "draw an image",
    )

    video_keywords = (
        "generate a video",
        "create a video",
        "make a video",
    )

    voice_keywords = (
        "voice command",
        "voice input",
        "voice message",
        "process this voice",
        "process voice",
        "speech input",
        "speech command",
        "speech to text",
        "transcribe this voice",
        "transcribe audio",
    )

    web_keywords = (
        "latest",
        "current",
        "today",
        "recent",
        "news",
    )

    if any(keyword in message for keyword in image_keywords):
        return Capability.IMAGE

    if any(keyword in message for keyword in video_keywords):
        return Capability.VIDEO

    if any(keyword in message for keyword in voice_keywords):
        return Capability.VOICE

    if any(keyword in message for keyword in web_keywords):
        return Capability.WEB

    return Capability.LLM