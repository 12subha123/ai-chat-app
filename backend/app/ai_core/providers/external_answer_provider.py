from abc import abstractmethod
from typing import Any

from ..answer_generation_provider import AnswerGenerationProvider


class ExternalAnswerGenerationProvider(
    AnswerGenerationProvider
):
    """
    Base interface for external LLM providers.

    Concrete providers such as OpenAI, Gemini, Claude,
    or other external services can implement this class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "external",
        }