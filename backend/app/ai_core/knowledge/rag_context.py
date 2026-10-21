from .knowledge_service import KnowledgeService


class RAGContextBuilder:
    """
    Retrieves relevant knowledge and formats it as context
    that can be supplied to the language model.
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService | None = None,
        top_k: int = 3,
    ):
        self.knowledge_service = (
            knowledge_service
            if knowledge_service is not None
            else KnowledgeService()
        )
        self.top_k = top_k

    def build_context(self, query: str) -> str:
        """
        Retrieve relevant knowledge for a user query and
        format it into a single context block.
        """
        if not query.strip():
            return ""

        results = self.knowledge_service.search(
            query=query,
            top_k=self.top_k,
        )

        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(results, start=1):
            text = result["text"].strip()

            if not text:
                continue

            context_parts.append(
                f"[Knowledge {index}]\n{text}"
            )

        return "\n\n".join(context_parts)