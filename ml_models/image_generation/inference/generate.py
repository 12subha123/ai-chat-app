from pathlib import Path

import torch
from PIL import Image

from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.reverse_diffusion import ReverseDiffusionProcess


def generate_image(
    checkpoint_path: str,
    output_path: str,
    image_size: int = 64,
):
    device = torch.device("cpu")

    model = SimpleDiffusionModel().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    diffusion = ReverseDiffusionProcess()

    image = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    for timestep in range(
        diffusion.timesteps - 1,
        -1,
        -1,
    ):
        timestep_tensor = torch.tensor(
            [timestep],
            device=device,
            dtype=torch.long,
        )

        image = diffusion.step(
            model,
            image,
            timestep_tensor,
        )

    image = image.squeeze(0)

    image = torch.clamp(
        image,
        -1.0,
        1.0,
    )

    image = (
        (image + 1.0)
        * 127.5
    )

    image = image.byte()

    image = image.permute(
        1,
        2,
        0,
    )

    image = Image.fromarray(
        image.cpu().numpy()
    )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output)

    print(
        f"Generated image saved: {output}"
    )


if __name__ == "__main__":
    generate_image(
        checkpoint_path=(
            "ml_models/image_generation/"
            "checkpoints/diffusion_epoch_1.pt"
        ),
        output_path=(
            "ml_models/image_generation/"
            "inference/generated_epoch_1.png"
        ),
    )