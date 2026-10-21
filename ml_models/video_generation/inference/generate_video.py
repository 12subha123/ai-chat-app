from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "ml_models"),
)

from video_generation.model.video_autoencoder import VideoAutoencoder


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "checkpoints"
    / "video_autoencoder_epoch_5.pt"
)

INPUT_CLIP = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "data"
    / "train"
    / "clip_00000.npy"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "inference"
    / "reconstructed_frames"
)


def main():
    device = torch.device("cpu")

    model = VideoAutoencoder().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    video = np.load(INPUT_CLIP)

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

    video = video.unsqueeze(0)

    video = video.to(device)

    with torch.no_grad():
        reconstructed = model(video)

    reconstructed = reconstructed.squeeze(0)

    reconstructed = reconstructed.permute(
        1,
        2,
        3,
        0,
    )

    reconstructed = (
        reconstructed
        .clamp(0.0, 1.0)
        * 255.0
    )

    reconstructed = (
        reconstructed
        .byte()
        .cpu()
        .numpy()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for frame_index, frame in enumerate(
        reconstructed
    ):
        output_path = (
            OUTPUT_DIRECTORY
            / f"frame_{frame_index:02d}.png"
        )

        Image.fromarray(
            frame
        ).save(output_path)

    print(
        f"Reconstructed {len(reconstructed)} frames."
    )

    print(
        f"Frames saved to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
