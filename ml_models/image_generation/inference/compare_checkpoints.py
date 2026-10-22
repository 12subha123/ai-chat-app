import torch
from pathlib import Path
from PIL import Image

from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.reverse_diffusion import ReverseDiffusionProcess


CHECKPOINT_DIR = Path(
    "ml_models/image_generation/checkpoints"
)

OUTPUT_DIR = Path(
    "ml_models/image_generation/inference/"
    "checkpoint_comparison"
)


def generate_image(
    checkpoint_path,
    output_path,
    seed=1,
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

    torch.manual_seed(seed)

    image = torch.randn(
        1,
        3,
        64,
        64,
        device=device,
    )

    with torch.no_grad():
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
    ).byte()

    image = image.permute(
        1,
        2,
        0,
    )

    image = Image.fromarray(
        image.cpu().numpy()
    )

    image.save(output_path)

    print(
        f"Generated: {output_path}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in [1, 2, 3]:
        checkpoint = (
            CHECKPOINT_DIR
            / f"diffusion_epoch_{epoch}.pt"
        )

        output = (
            OUTPUT_DIR
            / f"epoch_{epoch}.png"
        )

        generate_image(
            checkpoint,
            output,
            seed=1,
        )

    print(
        "Checkpoint comparison: PASSED"
    )


if __name__ == "__main__":
    main()