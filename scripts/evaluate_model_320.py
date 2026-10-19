import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.inference.inference_engine import InferenceEngine


TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints_expanded_320"
    / "checkpoint_epoch_20.pt"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_merged_320.json"
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


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def normalize_text(text):
    return (
        text.lower()
        .strip()
        .split()
    )


def token_overlap_score(
    prediction,
    reference,
):
    prediction_tokens = normalize_text(
        prediction
    )

    reference_tokens = normalize_text(
        reference
    )

    if not reference_tokens:
        return (
            1.0
            if not prediction_tokens
            else 0.0
        )

    if not prediction_tokens:
        return 0.0

    reference_counts = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(
                token,
                0,
            )
            + 1
        )

    prediction_counts = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(
                token,
                0,
            )
            + 1
        )

    overlap = 0

    for token, count in prediction_counts.items():
        overlap += min(
            count,
            reference_counts.get(
                token,
                0,
            ),
        )

    return (
        overlap
        / len(reference_tokens)
    )


def token_f1_score(
    prediction,
    reference,
):
    prediction_tokens = normalize_text(
        prediction
    )

    reference_tokens = normalize_text(
        reference
    )

    if (
        not prediction_tokens
        or not reference_tokens
    ):
        return 0.0

    reference_counts = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(
                token,
                0,
            )
            + 1
        )

    prediction_counts = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(
                token,
                0,
            )
            + 1
        )

    overlap = 0

    for token, count in prediction_counts.items():
        overlap += min(
            count,
            reference_counts.get(
                token,
                0,
            ),
        )

    precision = (
        overlap
        / len(prediction_tokens)
    )

    recall = (
        overlap
        / len(reference_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def evaluate_dataset(
    engine,
    dataset,
    dataset_name,
):
    overlap_scores = []
    f1_scores = []

    print()
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        prompt = item["prompt"]
        reference = item["answer"]

        prediction = engine.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=1,
            top_p=1.0,
        )

        overlap = token_overlap_score(
            prediction,
            reference,
        )

        f1 = token_f1_score(
            prediction,
            reference,
        )

        overlap_scores.append(
            overlap
        )

        f1_scores.append(
            f1
        )

        print()
        print(
            f"[{index}/{len(dataset)}]"
        )

        print(
            f"Prompt: {prompt}"
        )

        print(
            f"Expected: {reference}"
        )

        print(
            f"Generated: {prediction}"
        )

        print(
            f"Overlap: {overlap * 100:.2f}%"
        )

        print(
            f"F1: {f1 * 100:.2f}%"
        )

    average_overlap = (
        sum(overlap_scores)
        / len(overlap_scores)
        if overlap_scores
        else 0.0
    )

    average_f1 = (
        sum(f1_scores)
        / len(f1_scores)
        if f1_scores
        else 0.0
    )

    print()
    print("-" * 70)

    print(
        f"{dataset_name} average token overlap: "
        f"{average_overlap * 100:.2f}%"
    )

    print(
        f"{dataset_name} average token F1: "
        f"{average_f1 * 100:.2f}%"
    )

    return (
        average_overlap,
        average_f1,
    )


def main():

    print("=" * 70)
    print("320-MODEL EVALUATION")
    print("=" * 70)

    required_files = [
        TOKENIZER_PATH,
        CHECKPOINT_PATH,
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]

    for path in required_files:

        if not path.exists():

            print(
                "ERROR: Required file not found:"
            )

            print(path)

            sys.exit(1)

    train_data = load_json(
        TRAIN_PATH
    )

    validation_data = load_json(
        VALIDATION_PATH
    )

    test_data = load_json(
        TEST_PATH
    )

    print()
    print(
        f"Training examples: "
        f"{len(train_data)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_data)}"
    )

    print(
        f"Test examples: "
        f"{len(test_data)}"
    )

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    engine = InferenceEngine(
        tokenizer_path=str(
            TOKENIZER_PATH
        ),
        checkpoint_path=str(
            CHECKPOINT_PATH
        ),
        context_length=512,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.0,
        bias=True,
    )

    train_results = evaluate_dataset(
        engine,
        train_data,
        "TRAINING DATA",
    )

    validation_results = evaluate_dataset(
        engine,
        validation_data,
        "VALIDATION DATA",
    )

    test_results = evaluate_dataset(
        engine,
        test_data,
        "TEST DATA",
    )

    print()
    print("=" * 70)
    print("320-MODEL EVALUATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Training overlap: "
        f"{train_results[0] * 100:.2f}%"
    )

    print(
        f"Training F1: "
        f"{train_results[1] * 100:.2f}%"
    )

    print()

    print(
        f"Validation overlap: "
        f"{validation_results[0] * 100:.2f}%"
    )

    print(
        f"Validation F1: "
        f"{validation_results[1] * 100:.2f}%"
    )

    print()

    print(
        f"Test overlap: "
        f"{test_results[0] * 100:.2f}%"
    )

    print(
        f"Test F1: "
        f"{test_results[1] * 100:.2f}%"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()