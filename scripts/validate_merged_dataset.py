import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ORIGINAL_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_original_180.json"
)

MERGED_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_merged_320.json"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("MERGED DATASET VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    for path in [
        ORIGINAL_PATH,
        MERGED_PATH,
        TOKENIZER_PATH,
    ]:
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    original = load_json(
        ORIGINAL_PATH
    )

    merged = load_json(
        MERGED_PATH
    )

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------

    from llm_model.tokenizer.bpe_tokenizer import (
        BPETokenizer
    )

    tokenizer = BPETokenizer()

    tokenizer.load(
        str(TOKENIZER_PATH)
    )

    vocab_size = len(
        tokenizer.token_to_id
    )

    print(
        f"Original examples: {len(original)}"
    )

    print(
        f"Merged examples:   {len(merged)}"
    )

    print(
        f"Tokenizer vocabulary: {vocab_size}"
    )

    print()

    # ---------------------------------------------------------
    # Expected dataset size
    # ---------------------------------------------------------

    if len(merged) != 320:
        print(
            "WARNING: Expected 320 examples, "
            f"but found {len(merged)}."
        )

    # ---------------------------------------------------------
    # Duplicate prompt check
    # ---------------------------------------------------------

    prompts = [
        item["prompt"].strip().lower()
        for item in merged
    ]

    duplicate_count = (
        len(prompts)
        - len(set(prompts))
    )

    print(
        f"Duplicate prompts: {duplicate_count}"
    )

    # ---------------------------------------------------------
    # Structure validation
    # ---------------------------------------------------------

    structure_valid = True

    for index, item in enumerate(
        merged,
        start=1,
    ):

        if not isinstance(item, dict):
            print(
                f"ERROR: Example {index} "
                "is not an object."
            )
            structure_valid = False
            continue

        required_fields = [
            "prompt",
            "answer",
            "token_ids",
        ]

        for field in required_fields:
            if field not in item:
                print(
                    f"ERROR: Example {index} "
                    f"missing '{field}'."
                )
                structure_valid = False

        if not isinstance(
            item.get("prompt"),
            str,
        ):
            print(
                f"ERROR: Example {index} "
                "prompt is not a string."
            )
            structure_valid = False

        if not isinstance(
            item.get("answer"),
            str,
        ):
            print(
                f"ERROR: Example {index} "
                "answer is not a string."
            )
            structure_valid = False

        if not isinstance(
            item.get("token_ids"),
            list,
        ):
            print(
                f"ERROR: Example {index} "
                "token_ids is not a list."
            )
            structure_valid = False

    if structure_valid:
        print(
            "Structure validation: PASSED"
        )
    else:
        print(
            "Structure validation: FAILED"
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # Token ID validation
    # ---------------------------------------------------------

    token_ids_valid = True

    for index, item in enumerate(
        merged,
        start=1,
    ):

        prompt = item["prompt"].strip()
        answer = item["answer"].strip()

        full_text = (
            "User: "
            + prompt
            + "\nAssistant: "
            + answer
        )

        expected_token_ids = (
            tokenizer.encode(
                full_text,
                add_special_tokens=True,
            )
        )

        stored_token_ids = item[
            "token_ids"
        ]

        if stored_token_ids != expected_token_ids:
            print(
                f"ERROR: Token IDs do not match "
                f"for example {index}."
            )
            token_ids_valid = False

    if token_ids_valid:
        print(
            "Token ID validation: PASSED"
        )
    else:
        print(
            "Token ID validation: FAILED"
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # Original dataset preservation
    # ---------------------------------------------------------

    preservation_valid = True

    if len(merged) < len(original):
        preservation_valid = False
    else:
        for index, original_item in enumerate(
            original
        ):

            merged_item = merged[index]

            if (
                original_item.get("prompt")
                != merged_item.get("prompt")
            ):
                preservation_valid = False
                break

            if (
                original_item.get("answer")
                != merged_item.get("answer")
            ):
                preservation_valid = False
                break

            if (
                original_item.get("token_ids")
                != merged_item.get("token_ids")
            ):
                preservation_valid = False
                break

    if preservation_valid:
        print(
            "Original dataset preservation: PASSED"
        )
    else:
        print(
            "Original dataset preservation: FAILED"
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # Final duplicate check
    # ---------------------------------------------------------

    if duplicate_count == 0:
        print(
            "Duplicate check: PASSED"
        )
    else:
        print(
            "Duplicate check: WARNING"
        )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()