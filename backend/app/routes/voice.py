from fastapi import APIRouter, File, UploadFile

from app.ai_core.provider_factory import ProviderFactory


router = APIRouter(
    prefix="/voice",
    tags=["Voice"],
)


_provider_factory = ProviderFactory()
_provider_factory.register_default_providers()


@router.post("/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
):
    """
    Transcribe an uploaded audio file using
    the configured local voice provider.
    """

    audio_data = await file.read()

    if not audio_data:
        return {
            "message": "Audio file is empty.",
            "status": "error",
        }

    provider = _provider_factory.get_provider("voice")

    text = provider.transcribe(audio_data)

    return {
        "text": text,
        "provider": provider.name,
        "status": "success",
    }