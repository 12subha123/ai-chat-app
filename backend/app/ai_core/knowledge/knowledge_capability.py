from ..capability import AICapability
from ..models import AIRequest, AIResponse
from ..provider_factory import ProviderFactory
from ..tools.rag_llm_tool import RAGLLMTool


class KnowledgeCapability(AICapability):
    """
    Capability responsible for retrieving knowledge
    and generating an answer using the configured
    knowledge provider.
    """

    def __init__(
        self,
        rag_tool: RAGLLMTool | None = None,
        provider_factory: ProviderFactory | None = None,
    ):
        self.provider_factory = provider_factory

        self.rag_tool = rag_tool

        if self.rag_tool is None and self.provider_factory is None:
            self.rag_tool = RAGLLMTool()

    @property
    def name(self) -> str:
        return "knowledge"

    def can_handle(self, request: AIRequest) -> bool:
        return bool(request.message.strip())

    def execute(self, request: AIRequest) -> AIResponse:
        if self.provider_factory is not None:
            provider = self.provider_factory.get_provider("knowledge")
        else:
            provider = self.rag_tool

        response = provider.generate(
            message=request.message,
            conversation_context=request.context,
        )

        return AIResponse(
            message=response,
            source=self.name,
            metadata={
                "configured_provider": (
                    self.provider_factory.get_provider_name("knowledge")
                    if self.provider_factory is not None
                    else "internal_rag"
                ),
                "generation_provider": (
                    provider.generation_provider.name
                    if hasattr(provider, "generation_provider")
                    else "unknown"
                ),
            },
        )