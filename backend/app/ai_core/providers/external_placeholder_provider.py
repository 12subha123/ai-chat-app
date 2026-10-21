from typing import Any

from ..external_provider import ExternalProvider
from ..external_provider_config import ExternalProviderConfig


class ExternalPlaceholderProvider(ExternalProvider):
    """
    Placeholder for a future external API provider.
    """

    def __init__(self):
        super().__init__(
            config=ExternalProviderConfig(
                name="external_placeholder",
                api_key_env="EXTERNAL_PLACEHOLDER_API_KEY",
            )
        )

    @property
    def name(self) -> str:
        return "external_placeholder"

    def execute(self, request: Any) -> Any:
        return {
            "status": "not_connected",
            "provider": self.name,
            "message": "External provider is not connected yet.",
        }