import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.model.feedforward import FeedForward


def main():
    batch_size = 2
    sequence_length = 10
    embedding_dim = 256
    feed_forward_dim = 1024

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    feed_forward = FeedForward(
        embedding_dim=embedding_dim,
        feed_forward_dim=feed_forward_dim,
        dropout=0.0,
    )

    output = feed_forward(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    print("Feed-forward test passed!")


if __name__ == "__main__":
    main()