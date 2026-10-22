from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset loader for our image-generation model.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

        self.image_paths = sorted(
            path
            for path in self.data_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            )
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        image = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32,
        ).reshape(
            self.image_size,
            self.image_size,
            3,
        )

        image = image.permute(2, 0, 1)

        image = image / 255.0

        image = (image - 0.5) / 0.5

        return image