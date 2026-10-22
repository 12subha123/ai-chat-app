import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        half_dim = self.embedding_dim // 2

        scale = torch.log(
            torch.tensor(10000.0)
        ) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -scale
        )

        embeddings = (
            timesteps.float().unsqueeze(1)
            * embeddings.unsqueeze(0)
        )

        embeddings = torch.cat(
            [
                torch.sin(embeddings),
                torch.cos(embeddings),
            ],
            dim=1,
        )

        return embeddings


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            8,
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            8,
            out_channels,
        )

        self.time_projection = nn.Linear(
            time_embedding_dim,
            out_channels,
        )

        if in_channels != out_channels:
            self.residual_projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()

        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.residual_projection(x)

        x = self.conv1(x)

        x = self.norm1(x)

        x = self.activation(x)

        time = self.time_projection(
            time_embedding
        )

        time = time.unsqueeze(
            -1
        ).unsqueeze(-1)

        x = x + time

        x = self.conv2(x)

        x = self.norm2(x)

        x = self.activation(x)

        x = x + residual

        return x


class SimpleDiffusionModel(nn.Module):
    """
    Small U-Net-style diffusion model.

    The model predicts the noise added to an image
    during the forward diffusion process.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )

        self.input_block = ResidualBlock(
            image_channels,
            base_channels,
            time_embedding_dim,
        )

        self.downsample = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_block = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim,
        )

        self.middle_block = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim,
        )

        self.upsample = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.output_block = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_embedding_dim,
        )

        self.output = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        time_embedding = self.time_embedding(
            timesteps
        )

        x1 = self.input_block(
            images,
            time_embedding,
        )

        x2 = self.downsample(x1)

        x2 = self.down_block(
            x2,
            time_embedding,
        )

        x = self.middle_block(
            x2,
            time_embedding,
        )

        x = self.upsample(x)

        x = torch.cat(
            [
                x,
                x1,
            ],
            dim=1,
        )

        x = self.output_block(
            x,
            time_embedding,
        )

        x = self.output(x)

        return x