from typing import Any

from google import genai

from .external_answer_provider import ExternalAnswerGenerationProvider
from ..external_answer_provider_config import ExternalAnswerProviderConfig


class GeminiAnswerProvider(ExternalAnswerGenerationProvider):
    """
    Gemini-based external answer generation provider.

    This provider uses Google's Gemini Interactions API.
    """

    def __init__(self):
        self.config = ExternalAnswerProviderConfig(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            model_env="GEMINI_MODEL",
        )
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def available(self) -> bool:
        return self.config.configured

    @property
    def model(self) -> str:
        return self.config.get_model() or "gemini-3.8-flash"

    def _get_client(self):
        if not self.available:
            raise RuntimeError(
                "Gemini API key is not configured."
            )

        if self._client is None:
            api_key = self.config.get_api_key()

            self._client = genai.Client(
                api_key=api_key
            )

        return self._client

    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        if not prompt.strip():
            return ""

        client = self._get_client()

        interaction = client.interactions.create(
            model=self.model,
            input=prompt,
        )

        response = getattr(
            interaction,
            "output_text",
            None,
        )

        if response is None:
            return ""

        return response.strip()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "external",
            "model": self.model,
        }