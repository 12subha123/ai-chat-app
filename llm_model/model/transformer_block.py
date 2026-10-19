import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.

    Contains:
    1. Layer normalization
    2. Causal multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. Feed-forward network
    6. Residual connection
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.feed_forward_norm = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        attention_input = self.attention_norm(x)

        x = x + self.attention(
            attention_input
        )

        feed_forward_input = self.feed_forward_norm(x)

        x = x + self.feed_forward(
            feed_forward_input
        )

        return x