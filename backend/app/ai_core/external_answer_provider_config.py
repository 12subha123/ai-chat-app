import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .external_provider_config import ExternalProviderConfig


load_dotenv()


@dataclass
class ExternalAnswerProviderConfig(ExternalProviderConfig):
    """
    Configuration for an external answer-generation provider.
    """

    model_env: str | None = None

    def get_model(self) -> str | None:
        if not self.model_env:
            return None

        value = os.getenv(self.model_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    def status(self) -> dict:
        status = super().status()

        status["model_env"] = self.model_env
        status["model_configured"] = self.get_model() is not None

        return status