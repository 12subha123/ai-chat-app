from pathlib import Path
import sys
import uuid

import torch
from PIL import Image


class LocalDiffusionProvider:
    name = "local_diffusion"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[4]
        self.ml_models_path = self.project_root / "ml_models"

        if str(self.ml_models_path) not in sys.path:
            sys.path.insert(0, str(self.ml_models_path))

        from image_generation.model.diffusion_model import SimpleDiffusionModel
        from image_generation.model.reverse_diffusion import ReverseDiffusionProcess

        self.SimpleDiffusionModel = SimpleDiffusionModel
        self.ReverseDiffusionProcess = ReverseDiffusionProcess

        self.checkpoint_path = (
            self.ml_models_path
            / "image_generation"
            / "checkpoints"
            / "diffusion_epoch_3.pt"
        )

        self.output_directory = (
            self.ml_models_path
            / "image_generation"
            / "inference"
            / "backend_generated"
        )

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def generate(self, image_size: int = 32) -> str:
        if not self.available:
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {self.checkpoint_path}"
            )

        device = torch.device("cpu")

        model = self.SimpleDiffusionModel().to(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        diffusion = self.ReverseDiffusionProcess()

        image = torch.randn(
            1,
            3,
            image_size,
            image_size,
            device=device,
        )

        for timestep in range(diffusion.timesteps - 1, -1, -1):
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

        image = torch.clamp(image.squeeze(0), -1.0, 1.0)
        image = ((image + 1.0) * 127.5).byte()
        image = image.permute(1, 2, 0)

        output_image = Image.fromarray(image.cpu().numpy())

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / f"generated_{uuid.uuid4().hex}.png"
        )

        output_image.save(output_path)

        return str(output_path)