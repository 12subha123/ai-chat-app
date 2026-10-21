from ..answer_generation_provider import AnswerGenerationProvider


class MockGenerationProvider(AnswerGenerationProvider):
    """
    Mock answer-generation provider used for testing.
    """

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        conversation_context: list | None = None,
    ) -> str:
        return "MOCK PROVIDER RESPONSE"