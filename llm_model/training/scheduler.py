import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    step_size: int = 1000,
    gamma: float = 0.95,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create a learning-rate scheduler.

    Every `step_size` scheduler steps, the learning
    rate is multiplied by `gamma`.
    """

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    return scheduler