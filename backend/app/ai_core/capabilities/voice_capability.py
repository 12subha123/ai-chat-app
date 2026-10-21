from ..capability import AICapability
from ..models import AIRequest, AIResponse
from ..provider_factory import ProviderFactory
from ..voice_provider import VoiceProvider


class VoiceCapability(AICapability):
    """
    Capability for voice processing through a registered
    voice provider.
    """

    def __init__(
        self,
        provider_factory: ProviderFactory | None = None,
    ):
        self.provider_factory = provider_factory

    @property
    def name(self) -> str:
        return "voice"

    @property
    def available(self) -> bool:
        if self.provider_factory is None:
            return False

        try:
            provider = self.provider_factory.get_provider("voice")
        except ValueError:
            return False

        return isinstance(provider, VoiceProvider) and provider.available

    @property
    def provider(self) -> str:
        if self.provider_factory is None:
            return "not_connected"

        return self.provider_factory.get_provider_name("voice")

    def can_handle(self, request: AIRequest) -> bool:
        return bool(request.message.strip())

    def execute(self, request: AIRequest) -> AIResponse:
        if self.provider_factory is None:
            return AIResponse(
                message="Voice provider is not connected.",
                source=self.name,
            )

        provider = self.provider_factory.get_provider("voice")

        if not isinstance(provider, VoiceProvider):
            return AIResponse(
                message="Configured voice provider is invalid.",
                source=self.name,
                metadata={
                    "provider": self.provider,
                    "available": False,
                },
            )

        if not provider.available:
            return AIResponse(
                message="Voice provider is not available.",
                source=self.name,
                metadata={
                    "provider": provider.name,
                    "available": False,
                },
            )

        return AIResponse(
            message=(
                "Voice provider is connected and ready for "
                "speech-to-text and text-to-speech processing."
            ),
            source=self.name,
            metadata={
                "provider": provider.name,
                "available": provider.available,
                "voice_operations": [
                    "transcribe",
                    "synthesize",
                ],
            },
        )