from typing import Any

from ..voice_provider import VoiceProvider
from ..voice_provider_config import VoiceProviderConfig


class MockVoiceProvider(VoiceProvider):
    """
    Mock provider used to test the voice architecture
    before connecting a real STT/TTS service.
    """

    def __init__(self):
        self.config = VoiceProviderConfig(
            name="mock_voice",
        )

    @property
    def name(self) -> str:
        return "mock_voice"

    @property
    def available(self) -> bool:
        return True

    def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""

        return "MOCK TRANSCRIBED VOICE INPUT"

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        return b"MOCK SYNTHESIZED AUDIO"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "mock",
            "stt": "mock",
            "tts": "mock",
        }