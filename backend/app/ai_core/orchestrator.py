from .capabilities.image_capability import ImageCapability
from .capabilities.llm_capability import LLMCapability
from .capabilities.video_capability import VideoCapability
from .capabilities.voice_capability import VoiceCapability
from .capabilities.web_capability import WebCapability
from .capability_registry import CapabilityRegistry
from .knowledge.knowledge_capability import KnowledgeCapability
from .models import AIRequest, AIResponse
from .provider_config import ProviderConfig
from .provider_factory import ProviderFactory
from .providers.external_web_provider import ExternalWebProvider
from .router import route_request


class AIOrchestrator:
    def __init__(
        self,
        llm_capability: LLMCapability | None = None,
        knowledge_capability: KnowledgeCapability | None = None,
        provider_config: ProviderConfig | None = None,
    ):
        self.registry = CapabilityRegistry(
            provider_config=provider_config,
        )

        self.provider_factory = ProviderFactory(
            config=self.registry.provider_config,
        )

        self.provider_factory.register_default_providers()

        self._register_external_providers()

        self.registry.register(
            llm_capability
            if llm_capability is not None
            else LLMCapability(
                generation_provider=self.provider_factory.get_provider("llm"),
            )
        )

        self.registry.register(
            knowledge_capability
            if knowledge_capability is not None
            else KnowledgeCapability(
                provider_factory=self.provider_factory,
            )
        )

        self.registry.register(
            WebCapability(
                provider_factory=self.provider_factory,
            )
        )

        self.registry.register(
            VoiceCapability(
                provider_factory=self.provider_factory,
            )
        )

        self.registry.register(ImageCapability(provider_factory=self.provider_factory,))
        self.registry.register(
    VideoCapability(
        provider_factory=self.provider_factory,
    )
)

    def _register_external_providers(self) -> None:
        """
        Register external providers that are currently implemented.
        """
        web_provider_name = self.provider_factory.get_provider_name("web")

        if web_provider_name == "external_web":
            self.provider_factory.register_external_provider(
                ExternalWebProvider()
            )

    def handle(self, request: AIRequest) -> AIResponse:
        capability = route_request(request)

        return self._execute_registered_capability(
            name=capability.value,
            request=request,
        )

    def _execute_registered_capability(
        self,
        name: str,
        request: AIRequest,
    ) -> AIResponse:
        capability = self.registry.get(name)

        if capability is None:
            return AIResponse(
                message=f"Capability '{name}' is not available.",
                source="orchestrator",
            )

        if not capability.available:
            return AIResponse(
                message=(
                    f"The '{name}' capability is recognized, "
                    "but its provider is not connected yet."
                ),
                source="orchestrator",
                metadata={
                    "capability": name,
                    "available": False,
                    "provider": capability.provider,
                },
            )

        return capability.execute(request)
