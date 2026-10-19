from pathlib import Path
import sys

import torch


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Import sampling functions
# --------------------------------------------------

from llm_model.inference.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def main():

    # Fake logits for testing
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    # Temperature test
    temperature_logits = apply_temperature(
        logits,
        temperature=0.8,
    )

    assert temperature_logits.shape == logits.shape

    # Top-k test
    top_k_logits = apply_top_k(
        logits,
        top_k=3,
    )

    assert top_k_logits.shape == logits.shape

    # Top-p test
    top_p_logits = apply_top_p(
        logits,
        top_p=0.9,
    )

    assert top_p_logits.shape == logits.shape

    # Sampling test
    next_token = sample_next_token(
        logits,
        temperature=0.8,
        top_k=3,
        top_p=0.9,
    )

    assert isinstance(
        next_token,
        int,
    )

    assert 0 <= next_token < logits.numel()

    print(
        "Temperature test passed!"
    )

    print(
        "Top-k test passed!"
    )

    print(
        "Top-p test passed!"
    )

    print(
        "Sampling test passed!"
    )


if __name__ == "__main__":
    main()