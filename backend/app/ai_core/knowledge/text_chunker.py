import re


class TextChunker:
    """
    Splits text into sentence-aware chunks.

    Chunks are built from complete sentences rather than cutting
    arbitrarily through the middle of a sentence.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences while preserving sentence content.
        """
        normalized_text = re.sub(r"\s+", " ", text).strip()

        if not normalized_text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            normalized_text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def split(self, text: str) -> list[str]:
        """
        Create chunks from complete sentences.

        The chunk_size remains a target rather than a hard boundary.
        A sentence is never cut in the middle merely to satisfy the
        character limit.
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if not current_sentences:
                current_sentences.append(sentence)
                current_length = sentence_length
                continue

            proposed_length = (
                current_length
                + 1
                + sentence_length
            )

            if proposed_length <= self.chunk_size:
                current_sentences.append(sentence)
                current_length = proposed_length
            else:
                chunks.append(" ".join(current_sentences))

                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):
                    candidate_length = (
                        overlap_length
                        + len(previous_sentence)
                        + (1 if overlap_sentences else 0)
                    )

                    if candidate_length > self.overlap:
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence,
                    )
                    overlap_length = candidate_length

                current_sentences = (
                    overlap_sentences + [sentence]
                )

                current_length = sum(
                    len(item)
                    for item in current_sentences
                ) + max(
                    0,
                    len(current_sentences) - 1,
                )

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks