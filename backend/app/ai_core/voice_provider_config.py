import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_FILE)


@dataclass
class VoiceProviderConfig:
    """
    Configuration for a voice processing provider.
    """

    name: str
    stt_api_key_env: str | None = None
    tts_api_key_env: str | None = None
    stt_model_env: str | None = None
    tts_model_env: str | None = None

    def get_stt_api_key(self) -> str | None:
        if not self.stt_api_key_env:
            return None

        value = os.getenv(self.stt_api_key_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    def get_tts_api_key(self) -> str | None:
        if not self.tts_api_key_env:
            return None

        value = os.getenv(self.tts_api_key_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    def get_stt_model(self) -> str | None:
        if not self.stt_model_env:
            return None

        value = os.getenv(self.stt_model_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    def get_tts_model(self) -> str | None:
        if not self.tts_model_env:
            return None

        value = os.getenv(self.tts_model_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    @property
    def configured(self) -> bool:
        stt_configured = (
            self.stt_api_key_env is None
            or self.get_stt_api_key() is not None
        )

        tts_configured = (
            self.tts_api_key_env is None
            or self.get_tts_api_key() is not None
        )

        return stt_configured and tts_configured

    def status(self) -> dict:
        return {
            "name": self.name,
            "configured": self.configured,
            "stt_api_key_env": self.stt_api_key_env,
            "tts_api_key_env": self.tts_api_key_env,
            "stt_model_env": self.stt_model_env,
            "tts_model_env": self.tts_model_env,
            "stt_model_configured": (
                self.get_stt_model() is not None
            ),
            "tts_model_configured": (
                self.get_tts_model() is not None
            ),
        }