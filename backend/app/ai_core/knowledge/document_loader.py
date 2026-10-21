from pathlib import Path


class DocumentLoader:
    """
    Loads supported knowledge documents into plain text.
    """

    def load_text(self, document_path: str | Path) -> str:
        """
        Read a text or Markdown document and return its contents.
        """
        path = Path(document_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge document not found: {path}"
            )

        if path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError(
                f"Unsupported document type: {path.suffix}"
            )

        return path.read_text(encoding="utf-8")