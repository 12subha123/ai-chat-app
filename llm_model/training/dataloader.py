import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .dataset import LanguageModelDataset


PAD_TOKEN_ID = 0


def collate_batch(batch):
    """
    Pad variable-length conversation examples
    so they can be processed as one batch.
    """

    input_ids = [
        item[0]
        for item in batch
    ]

    target_ids = [
        item[1]
        for item in batch
    ]

    loss_masks = [
        item[2]
        for item in batch
    ]

    padded_input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )

    padded_target_ids = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )

    padded_loss_masks = pad_sequence(
        loss_masks,
        batch_first=True,
        padding_value=False,
    )

    return (
        padded_input_ids,
        padded_target_ids,
        padded_loss_masks,
    )


def create_dataloader_from_file(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:

    dataset = LanguageModelDataset(
        data_path=data_path,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def create_dataloader(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:

    dataset = LanguageModelDataset(
        data_path=data_path,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )