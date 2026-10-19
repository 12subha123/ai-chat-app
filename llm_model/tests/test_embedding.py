import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.model.embedding import TokenEmbedding
from llm_model.model.positional_encoding import (
    SinusoidalPositionalEncoding,
)


def main():
    batch_size = 2
    sequence_length = 10
    vocab_size = 10000
    embedding_dim = 256
    max_sequence_length = 512

    token_ids = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    token_embedding = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    positional_encoding = SinusoidalPositionalEncoding(
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
    )

    embeddings = token_embedding(
        token_ids
    )

    output = positional_encoding(
        embeddings
    )

    print(
        "Token IDs shape:",
        token_ids.shape,
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Position-aware output shape:",
        output.shape,
    )

    assert embeddings.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert torch.isfinite(output).all()

    print(
        "Embedding and positional encoding test passed!"
    )


if __name__ == "__main__":
    main()