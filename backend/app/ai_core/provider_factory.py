from typing import Any

from .provider_config import ProviderConfig


class ProviderFactory:
    def __init__(self, config: ProviderConfig | None = None):
        self.config = config if config is not None else ProviderConfig()
        self.config.validate()
        self._providers: dict[str, Any] = {}
        self.register_default_providers()

    def register(self, name: str, provider: Any) -> None:
        if not name.strip():
            raise ValueError("Provider name cannot be empty.")

        self._providers[name] = provider

    def register_external_provider(self, provider: Any) -> None:
        provider_name = getattr(provider, "name", None)

        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError(
                "External provider must define a non-empty name."
            )

        self.register(provider_name, provider)

    def get_provider_name(self, capability: str) -> str:
        provider_name = getattr(self.config, capability, None)

        if provider_name is None:
            raise ValueError(
                f"No provider configuration found for capability: {capability}"
            )

        return provider_name

    def get_provider(self, capability: str) -> Any:
        provider_name = self.get_provider_name(capability)
        provider = self._providers.get(provider_name)

        if provider is None:
            raise ValueError(
                f"Provider '{provider_name}' is not registered "
                f"for capability '{capability}'."
            )

        return provider

    def register_default_providers(self) -> None:
        from .providers.custom_transformer_provider import (
            CustomTransformerProvider,
        )
        from .providers.gemini_answer_provider import GeminiAnswerProvider
        from .providers.local_voice_provider import LocalVoiceProvider
        from .providers.mock_voice_provider import MockVoiceProvider
        from .providers.tavily_web_provider import TavilyWebProvider
        from .providers.local_video_provider import LocalVideoProvider
        from .tools.rag_llm_tool import RAGLLMTool
        from .providers.local_diffusion_provider import LocalDiffusionProvider

        self.register(
            "custom_transformer",
            CustomTransformerProvider(),
        )

        self.register(
            "internal_rag",
            RAGLLMTool(),
        )

        self.register(
            "tavily",
            TavilyWebProvider(),
        )

        self.register(
            "gemini",
            GeminiAnswerProvider(),
        )

        self.register(
            "mock_voice",
            MockVoiceProvider(),
        )

        self.register(
            "local_voice",
            LocalVoiceProvider(),
        )

        self.register("local_diffusion", LocalDiffusionProvider())
        self.register("local_video", LocalVideoProvider())

    def register_provider_for_capability(
        self,
        capability: str,
        provider: Any,
    ) -> None:
        provider_name = self.get_provider_name(capability)
        self.register(provider_name, provider)

    def external_status(self) -> dict[str, dict[str, Any]]:
        result = {}

        for name, provider in self._providers.items():
            if not hasattr(provider, "available"):
                continue

            if not hasattr(provider, "metadata"):
                continue

            result[name] = {
                "available": provider.available,
                "metadata": provider.metadata,
            }

        return result

    def status(self) -> dict[str, dict[str, Any]]:
        capabilities = {
            "llm": self.config.llm,
            "web_answer": self.config.web_answer,
            "knowledge": self.config.knowledge,
            "web": self.config.web,
            "voice": self.config.voice,
            "image": self.config.image,
            "video": self.config.video,
        }

        result = {}

        for capability, provider_name in capabilities.items():
            result[capability] = {
                "provider": provider_name,
                "registered": provider_name in self._providers,
            }

        return result
