from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    load_checkpoint,
)

from llm_model.model.transformer import (
    TransformerLanguageModel,
)


def main():
    data_path = (
        PROJECT_ROOT
        / "llm_model"
        / "data"
        / "processed"
        / "train.txt"
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "llm_model"
        / "checkpoints"
        / "test_real_training.pt"
    )

    dataloader = create_dataloader_from_file(
        data_path=str(data_path),
        sequence_length=16,
        batch_size=2,
        shuffle=True,
    )

    model = TransformerLanguageModel(
        vocab_size=125,
        context_length=16,
        embedding_dim=256,
        num_layers=2,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.0,
        bias=True,
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.01,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        step_size=1000,
        gamma=0.95,
    )

    average_loss = train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        loss=average_loss,
        path=str(checkpoint_path),
    )

    print("Checkpoint saved.")

    loaded_model = TransformerLanguageModel(
        vocab_size=125,
        context_length=16,
        embedding_dim=256,
        num_layers=2,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.0,
        bias=True,
    )

    loaded_optimizer = create_optimizer(
        model=loaded_model,
        learning_rate=3e-4,
        weight_decay=0.01,
    )

    loaded_scheduler = create_scheduler(
        optimizer=loaded_optimizer,
        step_size=1000,
        gamma=0.95,
    )

    checkpoint = load_checkpoint(
        model=loaded_model,
        optimizer=loaded_optimizer,
        scheduler=loaded_scheduler,
        path=str(checkpoint_path),
    )

    print("Checkpoint loaded.")
    print("Loaded epoch:", checkpoint["epoch"])
    print("Loaded loss:", checkpoint["loss"])

    assert checkpoint["epoch"] == 1

    print("\nReal training checkpoint test passed!")


if __name__ == "__main__":
    main()