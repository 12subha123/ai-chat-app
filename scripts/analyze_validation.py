from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "validation"
    / "validation.json"
)


def classify_question(prompt: str) -> str:
    question = prompt.strip().lower()

    if question.startswith("why "):
        return "why"

    if question.startswith("how "):
        return "how"

    if " vs " in question or "difference between" in question:
        return "comparison"

    if "example" in question:
        return "example"

    if "relationship" in question or "related" in question:
        return "relationship"

    if "cause" in question or "causes" in question:
        return "cause_effect"

    if "sign" in question or "indicate" in question:
        return "diagnosis"

    if question.startswith("what is"):
        return "definition"

    return "other"


def main():
    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    print("=" * 70)
    print("VALIDATION QUESTION ANALYSIS")
    print("=" * 70)

    categories = {}

    for index, item in enumerate(data, start=1):
        prompt = item["prompt"]
        category = classify_question(prompt)

        categories.setdefault(category, []).append(prompt)

        print(
            f"{index:2}. [{category:14}] {prompt}"
        )

    print()
    print("=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)

    for category, prompts in sorted(
        categories.items(),
        key=lambda x: (-len(x[1]), x[0]),
    ):
        print(
            f"{category:14}: {len(prompts)} question(s)"
        )


if __name__ == "__main__":
    main()