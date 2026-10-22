import torch

from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.reverse_diffusion import ReverseDiffusionProcess


CHECKPOINT = (
    "ml_models/image_generation/"
    "checkpoints/diffusion_epoch_3.pt"
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

    image = torch.randn(
        1,
        3,
        64,
        64,
        device=device,
    )

    print(
        "Initial noise | "
        f"mean={image.mean().item():.4f} "
        f"std={image.std().item():.4f}"
    )

    checkpoints = {
        999,
        750,
        500,
        250,
        100,
        0,
    }

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

            if timestep in checkpoints:
                print(
                    f"t={timestep:3d} | "
                    f"mean={image.mean().item():.4f} | "
                    f"std={image.std().item():.4f} | "
                    f"min={image.min().item():.4f} | "
                    f"max={image.max().item():.4f}"
                )

    print("Sampling diagnostic: PASSED")


if __name__ == "__main__":
    main()