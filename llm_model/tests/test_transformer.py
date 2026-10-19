import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.model.transformer import (
    TransformerLanguageModel,
)


def main():
    batch_size = 2
    sequence_length = 10

    vocab_size = 10000
    context_length = 512

    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    feed_forward_dim = 1024

    dropout = 0.0

    token_ids = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        dropout=dropout,
    )

    logits = model(token_ids)

    print(
        "Input token IDs shape:",
        token_ids.shape,
    )

    print(
        "Output logits shape:",
        logits.shape,
    )

    print(
        "Total parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size,
    )

    assert torch.isfinite(logits).all()

    print(
        "Complete Transformer test passed!"
    )


if __name__ == "__main__":
    main()