from typing import Any

import pyttsx3
from faster_whisper import WhisperModel

from ..voice_provider import VoiceProvider


class LocalVoiceProvider(VoiceProvider):
    """
    Local voice provider using:
    - faster-whisper for speech-to-text
    - pyttsx3 for text-to-speech

    No API key or cloud billing is required.
    """

    def __init__(
        self,
        whisper_model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.whisper_model_size = whisper_model_size
        self.device = device
        self.compute_type = compute_type

        self._whisper_model: WhisperModel | None = None
        self._tts_engine: Any | None = None

    @property
    def name(self) -> str:
        return "local_voice"

    @property
    def available(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "type": "voice",
            "provider": "local",
            "speech_to_text": "faster-whisper",
            "text_to_speech": "pyttsx3",
            "model_size": self.whisper_model_size,
            "device": self.device,
            "compute_type": self.compute_type,
        }

    def _get_whisper_model(self) -> WhisperModel:
        if self._whisper_model is None:
            self._whisper_model = WhisperModel(
                self.whisper_model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._whisper_model

    def _get_tts_engine(self) -> Any:
        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()

        return self._tts_engine

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio bytes using faster-whisper.

        The current implementation expects audio_data to contain
        a supported audio file format.
        """

        import tempfile
        from pathlib import Path

        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".audio",
                delete=False,
            ) as temp_file:
                temp_file.write(audio_data)
                temp_path = Path(temp_file.name)

            model = self._get_whisper_model()

            segments, _ = model.transcribe(
                str(temp_path),
                beam_size=5,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            return text.strip()

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using the local system TTS engine.

        The current implementation returns synthesized audio
        as WAV bytes.
        """

        import tempfile
        from pathlib import Path

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        engine = self._get_tts_engine()

        output_path = Path(
            tempfile.gettempdir()
        ) / "ai_chat_app_tts.wav"

        engine.save_to_file(
            text,
            str(output_path),
        )
        engine.runAndWait()

        return output_path.read_bytes()