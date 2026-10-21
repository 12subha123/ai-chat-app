from typing import Any

from .external_answer_provider import (
    ExternalAnswerGenerationProvider,
)
from ..external_answer_provider_config import (
    ExternalAnswerProviderConfig,
)


class MockExternalAnswerProvider(
    ExternalAnswerGenerationProvider
):
    """
    Mock external LLM provider used for architecture testing.

    This provider does not call any external API.
    """

    def __init__(self):
        self.config = ExternalAnswerProviderConfig(
            name="mock_external",
            api_key_env="MOCK_EXTERNAL_API_KEY",
            model_env="MOCK_EXTERNAL_MODEL",
        )

    @property
    def name(self) -> str:
        return "mock_external"

    @property
    def available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        return "MOCK EXTERNAL LLM RESPONSE"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "external",
            "model": "mock-model",
        }