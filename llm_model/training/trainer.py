import torch
import torch.nn.functional as F


def calculate_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate next-token prediction loss.

    Only positions where loss_mask is True
    contribute to the loss.
    """

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    target_ids = target_ids.reshape(
        batch_size * sequence_length,
    )

    loss_mask = loss_mask.reshape(
        batch_size * sequence_length,
    )

    # --------------------------------------------------
    # Calculate loss for every token.
    # --------------------------------------------------

    token_losses = F.cross_entropy(
        logits,
        target_ids,
        reduction="none",
    )

    # --------------------------------------------------
    # Keep only the tokens selected by loss_mask.
    # --------------------------------------------------

    masked_losses = token_losses[
        loss_mask
    ]

    if masked_losses.numel() == 0:
        raise ValueError(
            "loss_mask contains no active tokens."
        )

    loss = masked_losses.mean()

    return loss


def train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Perform one training step.
    """

    model.train()

    optimizer.zero_grad()

    logits = model(input_ids)

    loss = calculate_loss(
        logits=logits,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )

    loss.backward()

    optimizer.step()

    return loss.item()


def train_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> float:
    """
    Train the model for one complete epoch.
    """

    model.train()

    total_loss = 0.0
    number_of_batches = 0

    for (
        input_ids,
        target_ids,
        loss_mask,
    ) in dataloader:

        loss = train_step(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            optimizer=optimizer,
        )

        if scheduler is not None:
            scheduler.step()

        total_loss += loss
        number_of_batches += 1

    if number_of_batches == 0:
        raise ValueError(
            "The DataLoader contains no batches."
        )

    average_loss = (
        total_loss / number_of_batches
    )

    return average_loss