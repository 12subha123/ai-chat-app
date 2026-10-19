from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_original_180.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "expanded_candidates.json"
)


def make_candidates(example):
    prompt = example["prompt"]
    answer = example["answer"]

    concept = prompt.strip()

    candidates = [
        {
            "prompt": concept,
            "answer": answer,
            "source": "original",
        }
    ]

    return candidates


def main():
    with TRAIN_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    candidates = []

    for example in data:
        candidates.extend(make_candidates(example))

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("EXPANDED DATASET CANDIDATE GENERATOR")
    print("=" * 60)
    print(f"Original examples: {len(data)}")
    print(f"Candidate examples: {len(candidates)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()