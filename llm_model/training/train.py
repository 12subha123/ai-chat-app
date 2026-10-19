from pathlib import Path
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.tokenizer.bpe_tokenizer import BPETokenizer
from llm_model.model.transformer import TransformerLanguageModel
from llm_model.training.dataloader import create_dataloader_from_file
from llm_model.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "config"
    / "training_config.yaml"
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "config"
    / "model_config.yaml"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(
        checkpoint_dir.glob("checkpoint_epoch_*.pt")
    )

    if not checkpoints:
        return None

    def get_epoch(path):
        try:
            return int(
                path.stem.split("_")[-1]
            )
        except ValueError:
            return -1

    checkpoints.sort(
        key=get_epoch
    )

    return checkpoints[-1]


def calculate_loss(
    logits,
    target_ids,
    loss_masks,
):
    vocab_size = logits.size(-1)

    logits = logits.reshape(
        -1,
        vocab_size,
    )

    target_ids = target_ids.reshape(
        -1
    )

    loss_masks = loss_masks.reshape(
        -1
    )

    token_losses = torch.nn.functional.cross_entropy(
        logits,
        target_ids,
        reduction="none",
    )

    masked_losses = token_losses[
        loss_masks
    ]

    if masked_losses.numel() == 0:
        return token_losses.mean()

    return masked_losses.mean()


def main():

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config = load_yaml(
        CONFIG_PATH
    )

    model_config = load_yaml(
        MODEL_CONFIG_PATH
    )

    training_config = config["training"]
    scheduler_config = config["scheduler"]
    checkpoint_config = config["checkpoint"]
    data_config = config.get(
        "data",
        {},
    )
    device_config = config["device"]

    model_params = model_config["model"]

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if (
        device_config["type"] == "auto"
        and torch.cuda.is_available()
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        f"Device: {device}"
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = BPETokenizer()

    tokenizer.load(
        str(TOKENIZER_PATH)
    )

    vocab_size = len(
        tokenizer.token_to_id
    )

    print(
        f"Vocabulary size: {vocab_size}"
    )

    # ---------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------

    train_file = data_config.get(
        "train_file",
        "llm_model/data/processed/train_merged_320.json",
    )

    train_path = (
        PROJECT_ROOT
        / train_file
    )

    if not train_path.exists():
        print(
            f"ERROR: Training dataset not found:\n"
            f"{train_path}"
        )
        sys.exit(1)

    dataloader = create_dataloader_from_file(
        data_path=str(train_path),
        batch_size=training_config[
            "batch_size"
        ],
        shuffle=True,
    )

    print(
        f"Training dataset: {train_path}"
    )

    print(
        f"Training examples: "
        f"{len(dataloader.dataset)}"
    )

    print(
        f"Training batches: "
        f"{len(dataloader)}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=model_params[
            "context_length"
        ],
        embedding_dim=model_params[
            "embedding_dim"
        ],
        num_layers=model_params[
            "num_layers"
        ],
        num_heads=model_params[
            "num_heads"
        ],
        feed_forward_dim=model_params[
            "feed_forward_dim"
        ],
        dropout=model_params[
            "dropout"
        ],
        bias=model_params[
            "bias"
        ],
    )

    model.to(device)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=training_config[
            "learning_rate"
        ],
        weight_decay=training_config[
            "weight_decay"
        ],
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    scheduler = StepLR(
        optimizer,
        step_size=scheduler_config[
            "step_size"
        ],
        gamma=scheduler_config[
            "gamma"
        ],
    )

    # ---------------------------------------------------------
    # Checkpoint directory
    # ---------------------------------------------------------

    checkpoint_dir = (
        PROJECT_ROOT
        / checkpoint_config[
            "directory"
        ]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Checkpoint directory: "
        f"{checkpoint_dir}"
    )

    # ---------------------------------------------------------
    # Resume checkpoint if available
    # ---------------------------------------------------------

    latest_checkpoint = (
        find_latest_checkpoint(
            checkpoint_dir
        )
    )

    start_epoch = 1

    if latest_checkpoint is not None:

        print()
        print(
            f"Resuming from checkpoint:"
        )
        print(
            latest_checkpoint
        )

        checkpoint = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=str(
                latest_checkpoint
            ),
        )

        completed_epoch = checkpoint[
            "epoch"
        ]

        previous_loss = checkpoint[
            "loss"
        ]

        start_epoch = (
            completed_epoch + 1
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

        print(
            f"Previous loss: "
            f"{previous_loss:.4f}"
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    total_epochs = training_config[
        "epochs"
    ]

    if start_epoch > total_epochs:

        print()
        print(
            "Training already completed."
        )

        return

    print()
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(
        start_epoch,
        total_epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        batch_count = 0

        for (
            input_ids,
            target_ids,
            loss_masks,
        ) in dataloader:

            input_ids = input_ids.to(
                device
            )

            target_ids = target_ids.to(
                device
            )

            loss_masks = loss_masks.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                input_ids
            )

            loss = calculate_loss(
                logits,
                target_ids,
                loss_masks,
            )

            loss.backward()

            optimizer.step()

            scheduler.step()

            total_loss += (
                loss.item()
            )

            batch_count += 1

        average_loss = (
            total_loss
            / batch_count
        )

        print(
            f"Epoch {epoch}/{total_epochs} "
            f"- Loss: {average_loss:.4f}"
        )

        # -----------------------------------------------------
        # Save checkpoint
        # -----------------------------------------------------

        if checkpoint_config[
            "save_every_epoch"
        ]:

            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_epoch_{epoch}.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=average_loss,
                path=str(
                    checkpoint_path
                ),
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()