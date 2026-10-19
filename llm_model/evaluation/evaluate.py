from pathlib import Path
import sys
import json
import math

import torch


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from llm_model.inference.inference_engine import (
    InferenceEngine,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints"
    / "checkpoint_epoch_10.pt"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "validation"
    / "validation.json"
)

TEST_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "test"
    / "test.json"
)


# --------------------------------------------------
# Load evaluation examples
# --------------------------------------------------

def load_examples(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(file)

    if not isinstance(examples, list):
        raise ValueError(
            f"{path} must contain a list."
        )

    return examples


# --------------------------------------------------
# Calculate token overlap
# --------------------------------------------------

def calculate_token_overlap(
    generated: str,
    expected: str,
    tokenizer,
) -> float:

    generated_ids = tokenizer.encode(
        generated,
        add_special_tokens=False,
    )

    expected_ids = tokenizer.encode(
        expected,
        add_special_tokens=False,
    )

    if not expected_ids:
        return 0.0

    generated_set = set(
        generated_ids
    )

    expected_set = set(
        expected_ids
    )

    overlap = (
        len(
            generated_set
            & expected_set
        )
        / len(expected_set)
    )

    return overlap


# --------------------------------------------------
# Evaluate dataset
# --------------------------------------------------

def evaluate_dataset(
    engine: InferenceEngine,
    examples: list[dict],
    dataset_name: str,
) -> None:

    print()
    print("=" * 60)
    print(f"{dataset_name.upper()} EVALUATION")
    print("=" * 60)

    total_overlap = 0.0

    for index, example in enumerate(
        examples,
        start=1,
    ):

        prompt = example[
            "prompt"
        ]

        expected = example[
            "answer"
        ]

        generated = engine.generate(
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )

        overlap = calculate_token_overlap(
            generated=generated,
            expected=expected,
            tokenizer=engine.tokenizer,
        )

        total_overlap += overlap

        print()
        print(f"Example {index}")
        print("-" * 40)
        print(f"Prompt:    {prompt}")
        print(f"Expected:  {expected}")
        print(f"Generated: {generated}")
        print(
            f"Token overlap: "
            f"{overlap:.2%}"
        )

    if examples:

        average_overlap = (
            total_overlap
            / len(examples)
        )

        print()
        print(
            f"Average token overlap: "
            f"{average_overlap:.2%}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("AI-CHAT-APP MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    engine = InferenceEngine(
        tokenizer_path=str(
            TOKENIZER_PATH
        ),
        checkpoint_path=str(
            CHECKPOINT_PATH
        ),
    )

    print(
        f"Vocabulary size: "
        f"{engine.vocab_size}"
    )

    print(
        f"Device: "
        f"{engine.device}"
    )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    validation_examples = (
        load_examples(
            VALIDATION_PATH
        )
    )

    test_examples = (
        load_examples(
            TEST_PATH
        )
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    evaluate_dataset(
        engine=engine,
        examples=validation_examples,
        dataset_name="validation",
    )

    evaluate_dataset(
        engine=engine,
        examples=test_examples,
        dataset_name="test",
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()