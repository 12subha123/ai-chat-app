from ..answer_generation_provider import AnswerGenerationProvider
from ..capability import AICapability
from ..models import AIRequest, AIResponse
from ..providers.custom_transformer_provider import CustomTransformerProvider


class LLMCapability(AICapability):
    """
    Capability responsible for generating answers
    using the configured answer-generation provider.
    """

    def __init__(
        self,
        generation_provider: AnswerGenerationProvider | None = None,
    ):
        self.generation_provider = (
            generation_provider
            if generation_provider is not None
            else CustomTransformerProvider()
        )

    @property
    def name(self) -> str:
        return "llm"

    @property
    def available(self) -> bool:
        return bool(self.generation_provider.available)

    @property
    def provider(self) -> str:
        return self.generation_provider.name

    def can_handle(self, request: AIRequest) -> bool:
        return bool(request.message.strip())

    def execute(self, request: AIRequest) -> AIResponse:
        response = self.generation_provider.generate(
            prompt=request.message,
            conversation_context=request.context,
        )

        return AIResponse(
            message=response,
            source=self.name,
            metadata={
                "generation_provider": self.generation_provider.name,
            },
        )