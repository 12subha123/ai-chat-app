import json
import sys
from pathlib import Path


def validate_expansion(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("EXPANSION DATASET VALIDATION")
    print("=" * 70)

    print(f"Total examples: {len(data)}")
    print()

    # ---------------------------------------------------------
    # Structure validation
    # ---------------------------------------------------------

    structure_valid = True

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            print(f"ERROR: Example {index} is not an object.")
            structure_valid = False
            continue

        if "prompt" not in item:
            print(f"ERROR: Example {index} missing 'prompt'.")
            structure_valid = False

        if "answer" not in item:
            print(f"ERROR: Example {index} missing 'answer'.")
            structure_valid = False

        if not isinstance(item.get("prompt"), str):
            print(f"ERROR: Example {index} prompt is not a string.")
            structure_valid = False

        if not isinstance(item.get("answer"), str):
            print(f"ERROR: Example {index} answer is not a string.")
            structure_valid = False

    if structure_valid:
        print("Structure validation: PASSED")
    else:
        print("Structure validation: FAILED")
        sys.exit(1)

    print()

    # ---------------------------------------------------------
    # Duplicate prompt validation
    # ---------------------------------------------------------

    prompts = [item["prompt"].strip().lower() for item in data]

    duplicates = len(prompts) - len(set(prompts))

    print(f"Duplicate prompts: {duplicates}")

    if duplicates != 0:
        print()
        print("Duplicate prompts found:")

        seen = set()

        for prompt in prompts:
            if prompt in seen:
                print(f"  - {prompt}")
            else:
                seen.add(prompt)

        sys.exit(1)

    print()
    print("=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/validate_expansion.py "
            "<dataset_path>"
        )
        sys.exit(1)

    validate_expansion(sys.argv[1])