from typing import Any

from ..external_provider import ExternalProvider
from ..external_provider_config import ExternalProviderConfig


class ExternalWebProvider(ExternalProvider):
    """
    External provider placeholder for web search.

    No real web-search API is connected yet.
    """

    def __init__(self):
        super().__init__(
            config=ExternalProviderConfig(
                name="external_web",
                api_key_env="WEB_API_KEY",
                base_url_env="WEB_API_BASE_URL",
            )
        )

    @property
    def name(self) -> str:
        return "external_web"

    def execute(self, request: Any) -> Any:
        return {
            "status": "not_connected",
            "provider": self.name,
            "message": "External web provider is not connected yet.",
        }