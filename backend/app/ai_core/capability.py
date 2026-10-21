from abc import ABC, abstractmethod

from .models import AIRequest, AIResponse


class AICapability(ABC):
    """
    Base interface for every AI capability.

    Each capability must provide:
    - a unique name
    - a way to determine whether it can handle a request
    - an execution method
    - a status description
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        """
        Whether the capability is currently available.

        Subclasses can override this when availability depends
        on an external service or configuration.
        """
        return True

    @property
    def provider(self) -> str:
        """
        Name of the provider powering this capability.
        """
        return "internal"

    @abstractmethod
    def can_handle(self, request: AIRequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: AIRequest) -> AIResponse:
        raise NotImplementedError

    def status(self) -> dict:
        """
        Return a structured status description.
        """
        return {
            "name": self.name,
            "available": self.available,
            "provider": self.provider,
        }