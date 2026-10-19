import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from .positional_encoding import (
    SinusoidalPositionalEncoding,
)
from .transformer_block import TransformerBlock


class TransformerLanguageModel(nn.Module):
    """
    Decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_encoding = (
            SinusoidalPositionalEncoding(
                embedding_dim=embedding_dim,
                max_sequence_length=context_length,
            )
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids:
            Tensor with shape:

            (batch_size, sequence_length)

        Returns
        -------
        logits:
            Tensor with shape:

            (batch_size, sequence_length, vocab_size)
        """

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Sequence length exceeds the model "
                "context length."
            )

        x = self.token_embedding(
            token_ids
        )

        x = self.position_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits