from .capability import AICapability
from .provider_config import ProviderConfig


class CapabilityRegistry:
    """
    Registry containing all capabilities available to the AI system.
    """

    def __init__(
        self,
        provider_config: ProviderConfig | None = None,
    ):
        self._capabilities: dict[str, AICapability] = {}
        self.provider_config = (
            provider_config
            if provider_config is not None
            else ProviderConfig()
        )

    def register(self, capability: AICapability) -> None:
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> AICapability | None:
        return self._capabilities.get(name)

    def all(self) -> list[AICapability]:
        return list(self._capabilities.values())

    def names(self) -> list[str]:
        """
        Return the names of all registered capabilities.
        """
        return list(self._capabilities.keys())

    def status(self) -> dict[str, dict]:
        """
        Return detailed status information for every
        registered capability.
        """
        return {
            capability.name: capability.status()
            for capability in self._capabilities.values()
        }