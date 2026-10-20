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
    """
    Decide which AI capability should handle the request.

    The first version intentionally uses simple rules.
    Later, this router can be replaced with a more advanced
    intent classifier or AI-based routing system.
    """

    message = request.message.strip().lower()

    if not message:
        return Capability.LLM

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

    if any(keyword in message for keyword in web_keywords):
        return Capability.WEB

    return Capability.LLM