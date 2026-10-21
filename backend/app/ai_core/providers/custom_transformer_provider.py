from ..answer_generation_provider import AnswerGenerationProvider
from ..tools.llm_tool import generate_llm_response


class CustomTransformerProvider(AnswerGenerationProvider):
    """
    Answer-generation provider backed by the project's
    frozen custom Transformer model.
    """

    @property
    def name(self) -> str:
        return "custom_transformer"

    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        return generate_llm_response(
            message=prompt,
            conversation_context=conversation_context,
        )