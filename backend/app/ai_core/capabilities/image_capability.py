from ..capability import AICapability
from ..models import AIRequest, AIResponse
from ..provider_factory import ProviderFactory


class ImageCapability(AICapability):
    def __init__(self, provider_factory: ProviderFactory):
        self.provider_factory = provider_factory

    @property
    def name(self) -> str:
        return "image"

    @property
    def provider(self) -> str:
        return self.provider_factory.get_provider_name("image")

    @property
    def available(self) -> bool:
        provider = self.provider_factory.get_provider("image")
        return bool(getattr(provider, "available", False))

    def can_handle(self, request: AIRequest) -> bool:
        message = request.message.lower()

        keywords = (
            "generate an image",
            "create an image",
            "make an image",
            "draw an image",
        )

        return any(keyword in message for keyword in keywords)

    def execute(self, request: AIRequest) -> AIResponse:
        provider = self.provider_factory.get_provider("image")

        image_path = provider.generate(image_size=32)

        return AIResponse(
            message=f"Image generated successfully: {image_path}",
            source="image",
            metadata={
                "capability": "image",
                "provider": self.provider,
                "image_path": image_path,
            },
        )