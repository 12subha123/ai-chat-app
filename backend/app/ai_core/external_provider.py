from abc import ABC, abstractmethod
from typing import Any

from .external_provider_config import ExternalProviderConfig


class ExternalProvider(ABC):
    """
    Base interface for providers that depend on
    an external service or API.
    """

    def __init__(
        self,
        config: ExternalProviderConfig,
    ):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the provider's unique name.
        """
        raise NotImplementedError

    @property
    def available(self) -> bool:
        """
        Return whether the external provider is
        currently configured and available.
        """
        return self.config.configured

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return safe provider diagnostic information.
        """
        return {
            "name": self.name,
            "available": self.available,
            "type": "external",
            "configuration": self.config.status(),
        }

    @abstractmethod
    def execute(self, request: Any) -> Any:
        """
        Execute the provider operation.
        """
        raise NotImplementedError