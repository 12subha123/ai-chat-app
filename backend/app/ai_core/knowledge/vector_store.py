from pathlib import Path

import chromadb


class VectorStore:
    """
    Stores and searches knowledge embeddings using ChromaDB.

    The rest of the AI system interacts with this class instead
    of accessing ChromaDB directly. This keeps the vector database
    replaceable in the future.
    """

    def __init__(
        self,
        storage_directory: str | Path = "knowledge/vector_db",
        collection_name: str = "knowledge",
    ):
        self.storage_directory = Path(storage_directory)

        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.storage_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(
        self,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """
        Add one document chunk and its embedding.
        """
        self.collection.upsert(
            ids=[document_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """
        Search for the most semantically similar knowledge chunks.
        """
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """
        Return the number of stored knowledge chunks.
        """
        return self.collection.count()