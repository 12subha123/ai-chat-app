import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_DATASET_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_merged_260.json"
)

EXPANSION_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "targeted_expansion_60.json"
)

OUTPUT_PATH = (
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


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():

    print("=" * 70)
    print("MERGING TARGETED DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check required files
    # ---------------------------------------------------------

    if not BASE_DATASET_PATH.exists():
        print(
            f"ERROR: Base dataset not found:\n"
            f"{BASE_DATASET_PATH}"
        )
        sys.exit(1)

    if not EXPANSION_PATH.exists():
        print(
            f"ERROR: Expansion dataset not found:\n"
            f"{EXPANSION_PATH}"
        )
        sys.exit(1)

    if not TOKENIZER_PATH.exists():
        print(
            f"ERROR: Tokenizer not found:\n"
            f"{TOKENIZER_PATH}"
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    base_dataset = load_json(
        BASE_DATASET_PATH
    )

    expansion = load_json(
        EXPANSION_PATH
    )

    print(
        f"Existing examples:  {len(base_dataset)}"
    )

    print(
        f"Targeted examples:   {len(expansion)}"
    )

    # ---------------------------------------------------------
    # Check for duplicate prompts
    # ---------------------------------------------------------

    existing_prompts = {
        item["prompt"].strip().lower()
        for item in base_dataset
    }

    new_examples = []

    for item in expansion:

        prompt = item["prompt"].strip()
        prompt_key = prompt.lower()

        if prompt_key not in existing_prompts:
            new_examples.append(item)

    print(
        "New examples after duplicate check: "
        f"{len(new_examples)}"
    )

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------

    print()
    print("Loading tokenizer...")

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
        f"Tokenizer vocabulary size: "
        f"{vocab_size}"
    )

    # ---------------------------------------------------------
    # Tokenize new examples
    # ---------------------------------------------------------

    print()
    print("Tokenizing new examples...")

    tokenized_examples = []

    for index, item in enumerate(
        new_examples,
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

        token_ids = tokenizer.encode(
            full_text,
            add_special_tokens=True,
        )

        tokenized_examples.append(
            {
                "prompt": prompt,
                "answer": answer,
                "token_ids": token_ids,
            }
        )

    # ---------------------------------------------------------
    # Build final dataset
    # ---------------------------------------------------------

    final_dataset = list(
        base_dataset
    )

    final_dataset.extend(
        tokenized_examples
    )

    # ---------------------------------------------------------
    # Final count safety check
    # ---------------------------------------------------------

    expected_count = (
        len(base_dataset)
        + len(tokenized_examples)
    )

    if len(final_dataset) != expected_count:
        print(
            "ERROR: Final dataset count "
            "does not match expected count."
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # Save final dataset
    # ---------------------------------------------------------

    save_json(
        final_dataset,
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # Completion report
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGE COMPLETED")
    print("=" * 70)

    print(
        f"Existing examples:  "
        f"{len(base_dataset)}"
    )

    print(
        f"New examples added: "
        f"{len(tokenized_examples)}"
    )

    print(
        f"Final examples:     "
        f"{len(final_dataset)}"
    )

    print()
    print("Output:")
    print(OUTPUT_PATH)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()