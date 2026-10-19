import json
import sys
from pathlib import Path

import torch


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

CHECKPOINT_260 = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints_expanded_260"
    / "checkpoint_epoch_20.pt"
)

CHECKPOINT_320 = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints_expanded_320"
    / "checkpoint_epoch_20.pt"
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


def normalize(text):
    return text.lower().strip().split()


def f1_score(prediction, reference):

    prediction_tokens = normalize(
        prediction
    )

    reference_tokens = normalize(
        reference
    )

    if not prediction_tokens or not reference_tokens:
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


def create_engine(checkpoint):

    return InferenceEngine(
        tokenizer_path=str(
            TOKENIZER_PATH
        ),
        checkpoint_path=str(
            checkpoint
        ),
        context_length=512,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.0,
        bias=True,
    )


def compare_dataset(
    dataset,
    dataset_name,
    engine_260,
    engine_320,
):

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    total_260 = 0.0
    total_320 = 0.0

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        prompt = item["prompt"]
        reference = item["answer"]

        prediction_260 = engine_260.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=1,
            top_p=1.0,
        )

        prediction_320 = engine_320.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=1,
            top_p=1.0,
        )

        score_260 = f1_score(
            prediction_260,
            reference,
        )

        score_320 = f1_score(
            prediction_320,
            reference,
        )

        total_260 += score_260
        total_320 += score_320

        difference = (
            score_320
            - score_260
        )

        print()
        print(
            f"[{index}/{len(dataset)}] "
            f"F1 260: {score_260 * 100:.2f}% | "
            f"F1 320: {score_320 * 100:.2f}% | "
            f"Change: {difference * 100:+.2f}%"
        )

        print(
            f"Question: {prompt}"
        )

        print(
            f"Expected: {reference}"
        )

        print()
        print(
            "260 MODEL:"
        )
        print(
            prediction_260
        )

        print()
        print(
            "320 MODEL:"
        )
        print(
            prediction_320
        )

    average_260 = (
        total_260
        / len(dataset)
    )

    average_320 = (
        total_320
        / len(dataset)
    )

    print()
    print("-" * 80)

    print(
        f"{dataset_name} 260 F1: "
        f"{average_260 * 100:.2f}%"
    )

    print(
        f"{dataset_name} 320 F1: "
        f"{average_320 * 100:.2f}%"
    )

    print(
        f"{dataset_name} change: "
        f"{(average_320 - average_260) * 100:+.2f}%"
    )


def main():

    print("=" * 80)
    print("260 vs 320 MODEL COMPARISON")
    print("=" * 80)

    required_files = [
        TOKENIZER_PATH,
        CHECKPOINT_260,
        CHECKPOINT_320,
        VALIDATION_PATH,
        TEST_PATH,
    ]

    for path in required_files:

        if not path.exists():

            print(
                f"ERROR: File not found:\n{path}"
            )

            sys.exit(1)

    validation_data = load_json(
        VALIDATION_PATH
    )

    test_data = load_json(
        TEST_PATH
    )

    print()
    print(
        f"Validation examples: "
        f"{len(validation_data)}"
    )

    print(
        f"Test examples: "
        f"{len(test_data)}"
    )

    print()
    print("Loading 260 model...")

    engine_260 = create_engine(
        CHECKPOINT_260
    )

    print("Loading 320 model...")

    engine_320 = create_engine(
        CHECKPOINT_320
    )

    compare_dataset(
        validation_data,
        "VALIDATION",
        engine_260,
        engine_320,
    )

    compare_dataset(
        test_data,
        "TEST",
        engine_260,
        engine_320,
    )

    print()
    print("=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()