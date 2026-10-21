from ..capability import AICapability
from ..models import AIRequest, AIResponse
from ..provider_factory import ProviderFactory
from ..tools.web_llm_tool import WebLLMTool


class WebCapability(AICapability):
    """
    Capability responsible for searching the web using Tavily
    and generating an answer using the configured LLM.
    """

    def __init__(
        self,
        provider_factory: ProviderFactory | None = None,
        web_llm_tool: WebLLMTool | None = None,
    ):
        self.provider_factory = provider_factory

        self.web_llm_tool = web_llm_tool

        if self.web_llm_tool is None and self.provider_factory is None:
            self.web_llm_tool = WebLLMTool()

    @property
    def name(self) -> str:
        return "web"

    @property
    def available(self) -> bool:
        if self.provider_factory is not None:
            try:
                provider = self.provider_factory.get_provider("web")
            except ValueError:
                return False

            return bool(getattr(provider, "available", False))

        return bool(
            self.web_llm_tool is not None
            and self.web_llm_tool.web_provider.available
        )

    @property
    def provider(self) -> str:
        if self.provider_factory is not None:
            return self.provider_factory.get_provider_name("web")

        return self.web_llm_tool.web_provider.name

    def can_handle(self, request: AIRequest) -> bool:
        return bool(request.message.strip())

    def execute(self, request: AIRequest) -> AIResponse:
        if self.provider_factory is not None:
            web_provider = self.provider_factory.get_provider("web")

            tool = WebLLMTool(
                web_provider=web_provider,
                generation_provider=self.provider_factory.get_provider("web_answer"),
            )
        else:
            tool = self.web_llm_tool

        response = tool.generate(
            message=request.message,
            conversation_context=request.context,
        )

        return AIResponse(
            message=response,
            source=self.name,
            metadata={
                "provider": self.provider,
                "generation_provider": (
                    tool.generation_provider.name
                ),
            },
        )