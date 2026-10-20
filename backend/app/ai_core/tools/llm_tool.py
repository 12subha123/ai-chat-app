from app.services.llm_service import generate_response


def generate_llm_response(
    message: str,
    conversation_context: list | None = None,
) -> str:
    """
    Send a user message to the existing stable LLM service.

    The AI Core uses this function instead of accessing the
    Transformer model directly.
    """

    return generate_response(
        message=message,
        conversation_context=conversation_context,
    )