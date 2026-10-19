from pathlib import Path
import sys
import json


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from llm_model.inference.inference_engine import (
    InferenceEngine,
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

# --------------------------------------------------
# NEW EXPANDED MODEL
# --------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints_expanded_260"
    / "checkpoint_epoch_20.pt"
)

# --------------------------------------------------
# Evaluation datasets
# --------------------------------------------------

TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_merged_260.json"
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

def load_examples(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n"
            f"{path}"
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
# Token overlap
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

    generated_set = set(generated_ids)
    expected_set = set(expected_ids)

    overlap = (
        len(
            generated_set
            & expected_set
        )
        / len(expected_set)
    )

    return overlap


# --------------------------------------------------
# Token F1
# --------------------------------------------------

def calculate_token_f1(
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

    if (
        not generated_ids
        or not expected_ids
    ):
        return 0.0

    generated_counts = {}

    for token_id in generated_ids:

        generated_counts[token_id] = (
            generated_counts.get(
                token_id,
                0,
            )
            + 1
        )

    expected_counts = {}

    for token_id in expected_ids:

        expected_counts[token_id] = (
            expected_counts.get(
                token_id,
                0,
            )
            + 1
        )

    common_tokens = 0

    for token_id in generated_counts:

        if token_id in expected_counts:

            common_tokens += min(
                generated_counts[token_id],
                expected_counts[token_id],
            )

    precision = (
        common_tokens
        / len(generated_ids)
    )

    recall = (
        common_tokens
        / len(expected_ids)
    )

    if precision + recall == 0:

        return 0.0

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
    )

    return f1


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
    print(
        f"{dataset_name.upper()} EVALUATION"
    )
    print("=" * 60)

    total_overlap = 0.0
    total_f1 = 0.0

    for index, example in enumerate(
        examples,
        start=1,
    ):

        prompt = example["prompt"]
        expected = example["answer"]

        generated = engine.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=1,
            top_p=1.0,
        )

        overlap = calculate_token_overlap(
            generated=generated,
            expected=expected,
            tokenizer=engine.tokenizer,
        )

        f1 = calculate_token_f1(
            generated=generated,
            expected=expected,
            tokenizer=engine.tokenizer,
        )

        total_overlap += overlap
        total_f1 += f1

        print()
        print(
            f"Example {index}"
        )
        print("-" * 40)

        print(
            f"Prompt:    {prompt}"
        )

        print(
            f"Expected:  {expected}"
        )

        print(
            f"Generated: {generated}"
        )

        print(
            f"Token Overlap: "
            f"{overlap:.2%}"
        )

        print(
            f"Token F1: "
            f"{f1:.2%}"
        )

    if examples:

        average_overlap = (
            total_overlap
            / len(examples)
        )

        average_f1 = (
            total_f1
            / len(examples)
        )

        print()
        print(
            f"Average token overlap: "
            f"{average_overlap:.2%}"
        )

        print(
            f"Average token F1: "
            f"{average_f1:.2%}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AI-CHAT-APP MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Check required files
    # --------------------------------------------------

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n"
            f"{TOKENIZER_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print()
    print(
        "Loading expanded Epoch 20 model..."
    )

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

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_examples = load_examples(
        TRAIN_PATH
    )

    validation_examples = load_examples(
        VALIDATION_PATH
    )

    test_examples = load_examples(
        TEST_PATH
    )

    print()
    print(
        f"Training examples: "
        f"{len(train_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_examples)}"
    )

    print(
        f"Test examples: "
        f"{len(test_examples)}"
    )

    # --------------------------------------------------
    # Training evaluation
    # --------------------------------------------------

    evaluate_dataset(
        engine=engine,
        examples=train_examples,
        dataset_name="training",
    )

    # --------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------

    evaluate_dataset(
        engine=engine,
        examples=validation_examples,
        dataset_name="validation",
    )

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------

    evaluate_dataset(
        engine=engine,
        examples=test_examples,
        dataset_name="test",
    )

    # --------------------------------------------------
    # Completed
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()