from app.ai_core.capability import AICapability
from app.ai_core.models import AIRequest, AIResponse


class VideoCapability(AICapability):
    @property
    def name(self) -> str:
        return "video"

    @property
    def provider(self) -> str:
        return "local_video"

    def __init__(self, provider_factory):
        self.provider_factory = provider_factory

    @property
    def available(self) -> bool:
        provider = self.provider_factory.get_provider("video")

        return provider.available

    def can_handle(
        self,
        request: AIRequest,
    ) -> bool:
        message = request.message.lower()

        keywords = [
            "generate a video",
            "create a video",
            "make a video",
            "generate video",
            "create video",
            "make video",
        ]

        return any(
            keyword in message
            for keyword in keywords
        )

    def execute(
        self,
        request: AIRequest,
    ) -> AIResponse:
        provider = self.provider_factory.get_provider("video")

        video_path = provider.generate()

        return AIResponse(
            message="Video generated successfully.",
            source="video",
            metadata={
                "video_path": video_path,
                "prompt": request.message,
            },
        )

