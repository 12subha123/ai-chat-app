from abc import ABC, abstractmethod


class BaseLanguageModel(ABC):
    """
    Base interface for our language model.

    Every language model used by the application
    must implement this interface.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate text from a prompt.
        """
        raise NotImplementedError