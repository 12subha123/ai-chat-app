import json
import sys
from pathlib import Path
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_PATH = (
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def similarity(a, b):
    return SequenceMatcher(
        None,
        a.lower().strip(),
        b.lower().strip(),
    ).ratio()


def main():

    if not ORIGINAL_PATH.exists():
        print(f"ERROR: Original dataset not found: {ORIGINAL_PATH}")
        sys.exit(1)

    if not EXPANSION_PATH.exists():
        print(f"ERROR: Expansion dataset not found: {EXPANSION_PATH}")
        sys.exit(1)

    original = load_json(ORIGINAL_PATH)
    expansion = load_json(EXPANSION_PATH)

    original_prompts = [
        item["prompt"].strip()
        for item in original
    ]

    expansion_prompts = [
        item["prompt"].strip()
        for item in expansion
    ]

    print("=" * 70)
    print("TARGETED EXPANSION OVERLAP CHECK")
    print("=" * 70)

    print(f"Existing training examples: {len(original)}")
    print(f"Targeted expansion examples: {len(expansion)}")
    print()

    # ---------------------------------------------------------
    # Exact duplicate check
    # ---------------------------------------------------------

    original_lower = {
        prompt.lower()
        for prompt in original_prompts
    }

    exact_duplicates = []

    for prompt in expansion_prompts:
        if prompt.lower() in original_lower:
            exact_duplicates.append(prompt)

    print(f"Exact duplicate prompts: {len(exact_duplicates)}")

    if exact_duplicates:
        print()
        print("EXACT DUPLICATES")
        print("-" * 70)

        for prompt in exact_duplicates:
            print(prompt)

    # ---------------------------------------------------------
    # High similarity check
    # ---------------------------------------------------------

    high_similarity = []

    for expansion_index, new_prompt in enumerate(
        expansion_prompts,
        start=1,
    ):
        best_similarity = 0.0
        best_original = ""

        for original_prompt in original_prompts:
            score = similarity(
                new_prompt,
                original_prompt,
            )

            if score > best_similarity:
                best_similarity = score
                best_original = original_prompt

        if best_similarity >= 0.50:
            high_similarity.append(
                (
                    expansion_index,
                    new_prompt,
                    best_original,
                    best_similarity,
                )
            )

    print()
    print(
        "High-similarity prompts (>= 50%): "
        f"{len(high_similarity)}"
    )

    if high_similarity:
        print()
        print("HIGH-SIMILARITY PROMPTS")
        print("-" * 70)

        for (
            expansion_index,
            new_prompt,
            original_prompt,
            score,
        ) in high_similarity:

            print()
            print(f"Expansion #{expansion_index}")
            print(f"New:      {new_prompt}")
            print(f"Original: {original_prompt}")
            print(f"Similarity: {score * 100:.2f}%")

    print()
    print("=" * 70)
    print("OVERLAP CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()