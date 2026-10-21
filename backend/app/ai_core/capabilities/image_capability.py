from ..capability import AICapability
from ..models import AIRequest, AIResponse


class ImageCapability(AICapability):
    """
    Capability reserved for image generation.
    """

    @property
    def name(self) -> str:
        return "image"

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
            message="Image generation is not connected yet.",
            source=self.name,
        )