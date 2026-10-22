import torch
from pathlib import Path
from PIL import Image

from image_generation.data.image_dataset import ImageDataset
from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.diffusion_process import DiffusionProcess


CHECKPOINT = (
    "ml_models/image_generation/"
    "checkpoints/diffusion_epoch_3.pt"
)

DATA_DIR = (
    "ml_models/image_generation/"
    "data/train"
)

OUTPUT_DIR = Path(
    "ml_models/image_generation/"
    "inference/predicted_clean"
)


def save_image(tensor, path):
    tensor = tensor.squeeze(0)

    tensor = torch.clamp(
        tensor,
        -1.0,
        1.0,
    )

    tensor = (
        (tensor + 1.0)
        * 127.5
    ).byte()

    tensor = tensor.permute(
        1,
        2,
        0,
    )

    image = Image.fromarray(
        tensor.cpu().numpy()
    )

    image.save(path)


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

    diffusion = DiffusionProcess()

    dataset = ImageDataset(
        DATA_DIR,
        image_size=64,
    )

    clean_image = dataset[0].unsqueeze(0).to(device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        clean_image,
        OUTPUT_DIR / "original.png",
    )

    timesteps_to_test = [
        750,
        500,
        250,
        100,
    ]

    with torch.no_grad():

        for timestep in timesteps_to_test:

            timestep_tensor = torch.tensor(
                [timestep],
                dtype=torch.long,
                device=device,
            )

            torch.manual_seed(42)

            noisy_image, true_noise = (
                diffusion.add_noise(
                    clean_image,
                    timestep_tensor,
                )
            )

            predicted_noise = model(
                noisy_image,
                timestep_tensor,
            )

            alpha_cumprod = (
                diffusion.alpha_cumprod[
                    timestep
                ]
            )

            predicted_clean = (
                noisy_image
                - torch.sqrt(
                    1.0 - alpha_cumprod
                )
                * predicted_noise
            ) / torch.sqrt(
                alpha_cumprod
            )

            predicted_clean = torch.clamp(
                predicted_clean,
                -1.0,
                1.0,
            )

            save_image(
                noisy_image,
                OUTPUT_DIR
                / f"noisy_t{timestep}.png",
            )

            save_image(
                predicted_clean,
                OUTPUT_DIR
                / f"predicted_clean_t{timestep}.png",
            )

            mse = torch.mean(
                (
                    predicted_noise
                    - true_noise
                ) ** 2
            )

            print(
                f"t={timestep} | "
                f"noise prediction MSE="
                f"{mse.item():.6f}"
            )

    print(
        "Predicted-clean diagnostic: PASSED"
    )


if __name__ == "__main__":
    main()