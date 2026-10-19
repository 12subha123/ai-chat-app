import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.

    The model can attend to the current token and
    previous tokens, but never to future tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:

            (batch_size, sequence_length, embedding_dim)

        returns:

            (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, _ = x.shape

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        output = self.output_projection(
            attention_output
        )

        return output