import torch


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than 0."
        )

    return logits / temperature


def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:

    if top_k <= 0:
        return logits

    top_k = min(
        top_k,
        logits.size(-1),
    )

    values, _ = torch.topk(
        logits,
        top_k,
    )

    minimum_value = values[
        ...,
        -1,
        None,
    ]

    return torch.where(
        logits < minimum_value,
        torch.full_like(
            logits,
            float("-inf"),
        ),
        logits,
    )


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:

    if not 0 < top_p <= 1:
        raise ValueError(
            "top_p must be between 0 and 1."
        )

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        probabilities,
        dim=-1,
    )

    remove_mask = (
        cumulative_probabilities > top_p
    )

    remove_mask[..., 1:] = (
        remove_mask[..., :-1].clone()
    )

    remove_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove_mask,
        float("-inf"),
    )

    filtered_logits = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered_logits.scatter_(
        -1,
        sorted_indices,
        sorted_logits,
    )

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:

    logits = apply_temperature(
        logits,
        temperature,
    )

    logits = apply_top_k(
        logits,
        top_k,
    )

    logits = apply_top_p(
        logits,
        top_p,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return next_token.item()