from abc import ABC, abstractmethod


class GenerationProvider(ABC):
    """
    Common interface for all answer-generation systems.

    A generation provider receives a prompt and produces
    a text response.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of this generation provider.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        """
        Generate a response from the supplied prompt.
        """
        raise NotImplementedError