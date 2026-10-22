import torch
from pathlib import Path
from PIL import Image

from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.reverse_diffusion import ReverseDiffusionProcess


CHECKPOINT = (
    "ml_models/image_generation/"
    "checkpoints/diffusion_epoch_3.pt"
)

OUTPUT_DIR = Path(
    "ml_models/image_generation/"
    "inference/multi_samples"
)


def generate_sample(
    model,
    diffusion,
    seed,
    output_path,
):
    torch.manual_seed(seed)

    image = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        for timestep in range(
            diffusion.timesteps - 1,
            -1,
            -1,
        ):
            timestep_tensor = torch.tensor(
                [timestep],
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
        image.numpy()
    )

    image.save(output_path)

    print(
        f"Seed {seed}: {output_path}"
    )


def main():
    device = torch.device("cpu")

    model = SimpleDiffusionModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    diffusion = ReverseDiffusionProcess()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = [1, 2, 3, 4]

    for seed in seeds:
        output_path = (
            OUTPUT_DIR
            / f"sample_seed_{seed}.png"
        )

        generate_sample(
            model,
            diffusion,
            seed,
            output_path,
        )

    print("Multi-sample generation: PASSED")


if __name__ == "__main__":
    main()