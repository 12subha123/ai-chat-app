import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.model.attention import MultiHeadSelfAttention


def main():
    batch_size = 2
    sequence_length = 10
    embedding_dim = 256
    num_heads = 8

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.0,
    )

    output = attention(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    print("Attention test passed!")


if __name__ == "__main__":
    main()