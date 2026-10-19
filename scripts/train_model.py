from pathlib import Path
import sys

import torch

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from llm_model.model.transformer import (
    TransformerLanguageModel,
)

from llm_model.training.dataloader import (
    create_dataloader_from_file,
)

from llm_model.training.optimizer import (
    create_optimizer,
)

from llm_model.training.scheduler import (
    create_scheduler,
)

from llm_model.training.trainer import (
    train_epoch,
)

from llm_model.training.checkpoint import (
    save_checkpoint,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints"
)

VOCAB_SIZE = 500
CONTEXT_LENGTH = 128
EMBEDDING_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
FEED_FORWARD_DIM = 1024
DROPOUT = 0.1

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01

SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.95


# --------------------------------------------------
# Main training function
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AI-CHAT-APP MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # Check training data
    # --------------------------------------------------

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_DATA_PATH}"
        )

    print(
        f"Training data: {TRAIN_DATA_PATH}"
    )

    # --------------------------------------------------
    # Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------

    dataloader = create_dataloader_from_file(
        data_path=str(TRAIN_DATA_PATH),
        
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(
        f"Training batches: {len(dataloader)}"
    )

    # --------------------------------------------------
    # Create Transformer model
    # --------------------------------------------------

    model = TransformerLanguageModel(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        dropout=DROPOUT,
        bias=True,
    )

    model.to(device)

    # --------------------------------------------------
    # Create optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Create scheduler
    # --------------------------------------------------

    scheduler = create_scheduler(
        optimizer=optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )

    # --------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        average_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"checkpoint_epoch_{epoch}.pt"
        )

        save_checkpoint(
            path=str(checkpoint_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=average_loss,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    # --------------------------------------------------
    # Training completed
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()