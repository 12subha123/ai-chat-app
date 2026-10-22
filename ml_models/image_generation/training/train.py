from pathlib import Path

import torch

from image_generation.model.diffusion_model import SimpleDiffusionModel
from image_generation.model.diffusion_process import DiffusionProcess
from image_generation.training.dataloader import create_dataloader
from image_generation.training.train_step import diffusion_training_step


def train(
    data_dir: str,
    checkpoint_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    image_size: int = 64,
    learning_rate: float = 1e-4,
    resume_checkpoint: str | None = None,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataloader = create_dataloader(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SimpleDiffusionModel().to(device)

    diffusion = DiffusionProcess()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    start_epoch = 0

    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        print(
            f"Resuming from epoch {start_epoch}"
        )

        print(
            f"Previous loss: "
            f"{checkpoint['loss']:.4f}"
        )

    checkpoint_directory = Path(checkpoint_dir)

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Images:", len(dataloader.dataset))
    print("Batches per epoch:", len(dataloader))
    print("Starting epoch:", start_epoch + 1)
    print("Final epoch:", epochs)

    for epoch in range(
        start_epoch,
        epochs,
    ):
        total_loss = 0.0

        for batch_index, images in enumerate(
            dataloader
        ):
            images = images.to(device)

            loss = diffusion_training_step(
                model=model,
                diffusion=diffusion,
                images=images,
                optimizer=optimizer,
            )

            total_loss += loss

            if (batch_index + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"| Batch {batch_index + 1}/"
                    f"{len(dataloader)} "
                    f"| Loss: {loss:.4f}"
                )

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"completed | Average loss: "
            f"{average_loss:.4f}"
        )

        checkpoint_file = (
            checkpoint_directory
            / f"diffusion_epoch_{epoch + 1}.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
            },
            checkpoint_file,
        )

        print(
            f"Checkpoint saved: "
            f"{checkpoint_file}"
        )


if __name__ == "__main__":
    train(
        data_dir=(
            "ml_models/image_generation/"
            "data/train"
        ),
        checkpoint_dir=(
            "ml_models/image_generation/"
            "checkpoints"
        ),
        epochs=3,
        batch_size=16,
        image_size=32,
        learning_rate=1e-4,
        resume_checkpoint=None,
    )