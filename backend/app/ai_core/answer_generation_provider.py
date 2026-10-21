from abc import ABC, abstractmethod


class AnswerGenerationProvider(ABC):
    """
    Common interface for any provider that generates
    a natural-language answer from a prompt.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        raise NotImplementedError

    def status(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
        }