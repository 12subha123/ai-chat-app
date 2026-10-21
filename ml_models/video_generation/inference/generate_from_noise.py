from pathlib import Path
import sys

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "ml_models"),
)

from video_generation.model.video_autoencoder import VideoAutoencoder
from video_generation.model.latent_generator import LatentVideoGenerator


AUTOENCODER_CHECKPOINT = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "checkpoints"
    / "video_autoencoder_epoch_5.pt"
)

GENERATOR_CHECKPOINT = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "checkpoints"
    / "latent_generator_epoch_10.pt"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "inference"
    / "generated_frames"
)


def main():
    device = torch.device("cpu")

    autoencoder = VideoAutoencoder().to(device)

    autoencoder_checkpoint = torch.load(
        AUTOENCODER_CHECKPOINT,
        map_location=device,
    )

    autoencoder.load_state_dict(
        autoencoder_checkpoint["model_state_dict"]
    )

    autoencoder.eval()

    generator = LatentVideoGenerator().to(device)

    generator_checkpoint = torch.load(
        GENERATOR_CHECKPOINT,
        map_location=device,
    )

    generator.load_state_dict(
        generator_checkpoint["model_state_dict"]
    )

    generator.eval()

    noise = torch.randn(
        1,
        64 * 1 * 8 * 8,
        device=device,
    )

    with torch.no_grad():
        latent = generator(noise)

        generated_video = autoencoder.decoder(
            latent
        )

    generated_video = (
        generated_video
        .clamp(0.0, 1.0)
        * 255.0
    )

    generated_video = (
        generated_video
        .byte()
        .squeeze(0)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for frame_index, frame in enumerate(
        generated_video
    ):
        output_path = (
            OUTPUT_DIRECTORY
            / f"generated_frame_{frame_index:02d}.png"
        )

        Image.fromarray(
            frame
        ).save(output_path)

    print(
        f"Generated {len(generated_video)} frames."
    )

    print(
        f"Frames saved to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
