import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Transformer feed-forward network.

    Expands the embedding dimension, applies a
    nonlinear activation, then projects it back.
    """

    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                feed_forward_dim,
                bias=bias,
            ),
            nn.GELU(),
            nn.Linear(
                feed_forward_dim,
                embedding_dim,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)