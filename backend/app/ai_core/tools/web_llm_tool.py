from ..answer_generation_provider import AnswerGenerationProvider
from ..providers.custom_transformer_provider import CustomTransformerProvider
from ..providers.tavily_web_provider import TavilyWebProvider


class WebLLMTool:
    """
    Searches the live web using Tavily, selects the most
    relevant results, and sends compact context to an
    answer-generation provider.
    """

    def __init__(
        self,
        web_provider: TavilyWebProvider | None = None,
        generation_provider: AnswerGenerationProvider | None = None,
    ):
        self.web_provider = (
            web_provider
            if web_provider is not None
            else TavilyWebProvider()
        )

        self.generation_provider = (
            generation_provider
            if generation_provider is not None
            else CustomTransformerProvider()
        )

    @staticmethod
    def _prepare_results(
        results: list[dict],
        max_results: int = 3,
        max_content_length: int = 700,
    ) -> list[dict]:
        """
        Rank results by Tavily relevance score and keep
        only compact, useful content.
        """

        valid_results = [
            result
            for result in results
            if result.get("content", "").strip()
        ]

        valid_results.sort(
            key=lambda result: result.get("score", 0),
            reverse=True,
        )

        prepared = []

        for result in valid_results[:max_results]:
            title = result.get("title", "Untitled").strip()
            url = result.get("url", "").strip()
            content = result.get("content", "").strip()

            if len(content) > max_content_length:
                content = content[:max_content_length].rsplit(
                    " ",
                    1,
                )[0] + "..."

            prepared.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": result.get("score", 0),
                }
            )

        return prepared

    def generate(
        self,
        message: str,
        conversation_context: list | None = None,
    ) -> str:
        if not message.strip():
            return "Please enter a message."

        search_result = self.web_provider.execute(
            type(
                "WebRequest",
                (),
                {"message": message},
            )()
        )

        results = search_result.get("results", [])

        prepared_results = self._prepare_results(results)

        if not prepared_results:
            return "I could not find relevant information on the web."

        context_parts = []

        for index, result in enumerate(prepared_results, start=1):
            context_parts.append(
                f"[Source {index}]\n"
                f"Title: {result['title']}\n"
                f"Content: {result['content']}"
            )

        web_context = "\n\n".join(context_parts)

        augmented_prompt = (
            "Answer the user's question using the web sources below.\n"
            "Use only information supported by these sources.\n"
            "Give a concise answer.\n\n"
            f"{web_context}\n\n"
            f"User question: {message}\n\n"
            "Answer:"
        )

        return self.generation_provider.generate(
            prompt=augmented_prompt,
            conversation_context=conversation_context,
        )