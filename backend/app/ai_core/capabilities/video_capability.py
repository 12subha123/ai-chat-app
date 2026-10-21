from ..capability import AICapability
from ..models import AIRequest, AIResponse


class VideoCapability(AICapability):
    """
    Capability reserved for video generation.
    """

    @property
    def name(self) -> str:
        return "video"

    @property
    def available(self) -> bool:
        return False

    @property
    def provider(self) -> str:
        return "not_connected"

    def can_handle(self, request: AIRequest) -> bool:
        return bool(request.message.strip())

    def execute(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            message="Video generation is not connected yet.",
            source=self.name,
        )