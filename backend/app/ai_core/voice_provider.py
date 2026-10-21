from abc import ABC, abstractmethod
from typing import Any


class VoiceProvider(ABC):
    """
    Base interface for voice processing providers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "voice",
        }

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Convert speech audio into text.
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text into speech audio.
        """
        raise NotImplementedError