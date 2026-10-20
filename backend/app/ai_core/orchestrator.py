from .models import AIRequest, AIResponse
from .router import Capability, route_request
from .tools.llm_tool import generate_llm_response


class AIOrchestrator:
    """
    Central coordinator for the AI system.

    The orchestrator receives a standardized AIRequest,
    determines the required capability, and delegates
    the request to the appropriate service.
    """

    def handle(self, request: AIRequest) -> AIResponse:
        capability = route_request(request)

        if capability == Capability.LLM:
            return self._handle_llm(request)

        if capability == Capability.KNOWLEDGE:
            return self._handle_knowledge(request)

        if capability == Capability.WEB:
            return self._handle_web(request)

        if capability == Capability.VOICE:
            return self._handle_voice(request)

        if capability == Capability.IMAGE:
            return self._handle_image(request)

        if capability == Capability.VIDEO:
            return self._handle_video(request)

        return AIResponse(
            message="I could not determine which capability to use.",
            source="orchestrator",
        )

    def _handle_llm(self, request: AIRequest) -> AIResponse:
        response = generate_llm_response(
            message=request.message,
            conversation_context=request.context,
        )

        return AIResponse(
            message=response,
            source="llm",
        )

    def _handle_knowledge(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Knowledge retrieval is not connected yet.",
            source="knowledge",
        )

    def _handle_web(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Web retrieval is not connected yet.",
            source="web",
        )

    def _handle_voice(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Voice processing is not connected yet.",
            source="voice",
        )

    def _handle_image(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Image generation is not connected yet.",
            source="image",
        )

    def _handle_video(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Video generation is not connected yet.",
            source="video",
        )