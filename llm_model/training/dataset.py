import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from llm_model.tokenizer.bpe_tokenizer import BPETokenizer


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class LanguageModelDataset(Dataset):

    def __init__(
        self,
        data_path: str,
    ):

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n"
                f"{self.data_path}"
            )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found:\n"
                f"{TOKENIZER_PATH}"
            )

        # --------------------------------------------------
        # Load dataset
        # --------------------------------------------------

        with self.data_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.data = json.load(file)

        if not isinstance(self.data, list):
            raise ValueError(
                "Dataset must contain a JSON list."
            )

        # --------------------------------------------------
        # Load tokenizer
        # --------------------------------------------------

        self.tokenizer = BPETokenizer()

        self.tokenizer.load(
            str(TOKENIZER_PATH)
        )

        # --------------------------------------------------
        # Prepare all examples once
        # --------------------------------------------------

        self.examples = []

        for index, example in enumerate(
            self.data
        ):

            prompt = example["prompt"]
            answer = example["answer"]

            stored_token_ids = example[
                "token_ids"
            ]

            # --------------------------------------------------
            # Reconstruct the exact training text
            # --------------------------------------------------

            full_text = (
                f"User: {prompt}\n"
                f"Assistant: {answer}"
            )

            prompt_text = (
                f"User: {prompt}\n"
                f"Assistant:"
            )

            # --------------------------------------------------
            # Tokenize
            # --------------------------------------------------

            prompt_token_ids = (
                self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                )
            )

            full_token_ids = (
                self.tokenizer.encode(
                    full_text,
                    add_special_tokens=True,
                )
            )

            # --------------------------------------------------
            # Verify stored token IDs
            # --------------------------------------------------

            if full_token_ids != stored_token_ids:
                raise ValueError(
                    "Stored token_ids do not match "
                    "the current tokenizer output.\n"
                    f"Example index: {index}\n"
                    f"Prompt: {prompt}"
                )

            # --------------------------------------------------
            # Input and target
            # --------------------------------------------------

            input_ids = full_token_ids[:-1]

            target_ids = full_token_ids[1:]

            # --------------------------------------------------
            # Loss mask
            #
            # Only calculate loss for the assistant answer
            # and EOS token.
            # --------------------------------------------------

            prompt_length = len(
                prompt_token_ids
            )

            loss_mask = [
                False
                for _ in target_ids
            ]

            for position in range(
                prompt_length,
                len(target_ids),
            ):

                loss_mask[position] = True

            # --------------------------------------------------
            # Convert to tensors
            # --------------------------------------------------

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
            )

            target_tensor = torch.tensor(
                target_ids,
                dtype=torch.long,
            )

            mask_tensor = torch.tensor(
                loss_mask,
                dtype=torch.bool,
            )

            # --------------------------------------------------
            # Store prepared example
            # --------------------------------------------------

            self.examples.append(
                (
                    input_tensor,
                    target_tensor,
                    mask_tensor,
                )
            )

        # --------------------------------------------------
        # Dataset summary
        # --------------------------------------------------

        print(
            f"Prepared {len(self.examples)} "
            f"training examples."
        )

    # --------------------------------------------------
    # Dataset length
    # --------------------------------------------------

    def __len__(self):

        return len(self.examples)

    # --------------------------------------------------
    # Get example
    # --------------------------------------------------

    def __getitem__(
        self,
        index,
    ):

        return self.examples[index]