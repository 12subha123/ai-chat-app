from torch.utils.data import DataLoader

from image_generation.data.image_dataset import ImageDataset


def create_dataloader(
    data_dir: str,
    image_size: int = 64,
    batch_size: int = 16,
    shuffle: bool = True,
):
    dataset = ImageDataset(
        data_dir=data_dir,
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return dataloader