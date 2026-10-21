import torch
import torch.nn as nn


class LatentVideoGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(64 * 1 * 8 * 8, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 64 * 1 * 8 * 8),
        )

    def forward(self, noise):
        batch_size = noise.shape[0]

        output = self.network(noise)

        output = output.view(
            batch_size,
            64,
            1,
            8,
            8,
        )

        return output
