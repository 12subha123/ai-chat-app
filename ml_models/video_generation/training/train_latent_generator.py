from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "ml_models"),
)

from video_generation.model.video_autoencoder import VideoAutoencoder
from video_generation.model.latent_generator import LatentVideoGenerator


DATA_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "data"
    / "train"
)

AUTOENCODER_CHECKPOINT = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "checkpoints"
    / "video_autoencoder_epoch_5.pt"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "checkpoints"
)

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.001


class VideoClipDataset(Dataset):
    def __init__(self, data_directory):
        self.files = sorted(
            Path(data_directory).glob("*.npy")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No training clips found in: {data_directory}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        video = np.load(
            self.files[index]
        )

        video = torch.from_numpy(
            video
        ).float()

        video = video / 255.0

        video = video.permute(
            3,
            0,
            1,
            2,
        )

        return video


def train():
    device = torch.device("cpu")

    dataset = VideoClipDataset(
        DATA_DIRECTORY
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    autoencoder = VideoAutoencoder().to(device)

    checkpoint = torch.load(
        AUTOENCODER_CHECKPOINT,
        map_location=device,
    )

    autoencoder.load_state_dict(
        checkpoint["model_state_dict"]
    )

    autoencoder.eval()

    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    generator = LatentVideoGenerator().to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.MSELoss()

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(f"Training clips: {len(dataset)}")
    print(f"Epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        generator.train()

        total_loss = 0.0

        for videos in dataloader:
            videos = videos.to(device)

            with torch.no_grad():
                target_latents = autoencoder.encoder(
                    videos
                )

            noise = torch.randn(
                videos.shape[0],
                64 * 1 * 8 * 8,
                device=device,
            )

            predicted_latents = generator(
                noise
            )

            loss = loss_function(
                predicted_latents,
                target_latents,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1} "
            f"average loss: {average_loss:.6f}"
        )

        checkpoint_path = (
            CHECKPOINT_DIRECTORY
            / f"latent_generator_epoch_{epoch + 1}.pt"
        )

        torch.save(
            {
                "model_state_dict": generator.state_dict(),
                "epoch": epoch + 1,
                "loss": average_loss,
            },
            checkpoint_path,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )


if __name__ == "__main__":
    train()
