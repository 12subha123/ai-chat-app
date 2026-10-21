from ..answer_generation_provider import AnswerGenerationProvider
from ..knowledge.rag_context import RAGContextBuilder
from ..providers.custom_transformer_provider import CustomTransformerProvider


class RAGLLMTool:
    """
    Retrieves relevant knowledge and sends the augmented
    prompt to an answer-generation provider.
    """

    def __init__(
        self,
        context_builder: RAGContextBuilder | None = None,
        generation_provider: AnswerGenerationProvider | None = None,
    ):
        self.context_builder = (
            context_builder
            if context_builder is not None
            else RAGContextBuilder()
        )

        self.generation_provider = (
            generation_provider
            if generation_provider is not None
            else CustomTransformerProvider()
        )

    def generate(
        self,
        message: str,
        conversation_context: list | None = None,
    ) -> str:
        knowledge_context = self.context_builder.build_context(message)

        if knowledge_context:
            augmented_prompt = (
                "Use the following knowledge to help answer "
                "the user's question.\n\n"
                f"{knowledge_context}\n\n"
                "User question:\n"
                f"{message}\n\n"
                "Answer the user's question clearly and accurately."
            )
        else:
            augmented_prompt = message

        return self.generation_provider.generate(
            prompt=augmented_prompt,
            conversation_context=conversation_context,
        )