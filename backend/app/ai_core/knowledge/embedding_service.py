from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Converts text into numerical vector embeddings.

    These embeddings will later be used by the vector database
    to find knowledge that is semantically similar to a user's query.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text into a 384-dimensional embedding.
        """
        if not text.strip():
            return []

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple texts into embeddings.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()