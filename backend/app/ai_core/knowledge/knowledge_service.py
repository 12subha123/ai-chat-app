from pathlib import Path

from .document_loader import DocumentLoader
from .embedding_service import EmbeddingService
from .text_chunker import TextChunker
from .vector_store import VectorStore


class KnowledgeService:
    """
    Coordinates the complete knowledge ingestion and retrieval pipeline.

    Pipeline:

        Document
            ↓
        DocumentLoader
            ↓
        TextChunker
            ↓
        EmbeddingService
            ↓
        VectorStore
    """

    def __init__(
        self,
        knowledge_directory: str | Path = "knowledge",
        vector_store_directory: str | Path = "knowledge/vector_db",
    ):
        self.knowledge_directory = Path(knowledge_directory)

        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=500,
            overlap=50,
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            storage_directory=vector_store_directory,
            collection_name="knowledge",
        )

    def list_documents(self) -> list[Path]:
        """
        Return all supported knowledge documents.
        """
        if not self.knowledge_directory.exists():
            return []

        supported_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".json",
        }

        return sorted(
            path
            for path in self.knowledge_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
            and "vector_db" not in path.parts
        )

    def ingest_document(self, document_path: str | Path) -> int:
        """
        Load, chunk, embed, and store one document.

        Returns the number of chunks stored.
        """
        path = Path(document_path)

        text = self.loader.load_text(path)
        chunks = self.chunker.split(text)

        if not chunks:
            return 0

        embeddings = self.embedding_service.embed_texts(chunks)

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            document_id = f"{path.stem}-chunk-{index}"

            self.vector_store.add(
                document_id=document_id,
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": str(path),
                    "chunk_index": index,
                },
            )

        return len(chunks)

    def ingest_all_documents(self) -> int:
        """
        Ingest every supported document in the knowledge directory.

        Returns the total number of chunks stored.
        """
        total_chunks = 0

        for document in self.list_documents():
            total_chunks += self.ingest_document(document)

        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the knowledge base using semantic similarity.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        matches = []

        for document, distance, metadata in zip(
            documents,
            distances,
            metadatas,
        ):
            matches.append(
                {
                    "text": document,
                    "distance": distance,
                    "metadata": metadata,
                }
            )

        return matches