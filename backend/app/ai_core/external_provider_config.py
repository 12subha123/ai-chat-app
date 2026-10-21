import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_FILE)


@dataclass
class ExternalProviderConfig:
    name: str
    api_key_env: str | None = None
    base_url_env: str | None = None

    def get_api_key(self) -> str | None:
        if not self.api_key_env:
            return None

        value = os.getenv(self.api_key_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    def get_base_url(self) -> str | None:
        if not self.base_url_env:
            return None

        value = os.getenv(self.base_url_env)

        if value is None:
            return None

        value = value.strip()

        return value if value else None

    @property
    def configured(self) -> bool:
        if self.api_key_env is None:
            return False

        return self.get_api_key() is not None

    def status(self) -> dict:
        return {
            "name": self.name,
            "configured": self.configured,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
        }