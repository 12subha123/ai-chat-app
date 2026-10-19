from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train.json"
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


STOPWORDS = {
    "what",
    "is",
    "are",
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "how",
    "why",
    "does",
    "do",
    "can",
    "be",
    "with",
    "from",
    "used",
}


def load_data(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def words(text):
    return {
        word
        for word in re.findall(r"[a-zA-Z]+", text.lower())
        if word not in STOPWORDS
    }


def similarity(prompt_a, prompt_b):
    words_a = words(prompt_a)
    words_b = words(prompt_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def find_best_match(prompt, train_data):
    best_prompt = ""
    best_score = 0.0

    for example in train_data:
        train_prompt = example["prompt"]
        score = similarity(prompt, train_prompt)

        if score > best_score:
            best_score = score
            best_prompt = train_prompt

    return best_prompt, best_score


def analyze_dataset(name, data, train_data):
    print()
    print("=" * 70)
    print(f"{name.upper()} → TRAINING QUESTION SIMILARITY")
    print("=" * 70)

    for index, example in enumerate(data, start=1):
        prompt = example["prompt"]

        best_prompt, score = find_best_match(
            prompt,
            train_data,
        )

        print()
        print(f"{index:2}. {prompt}")
        print(f"    Closest training question: {best_prompt}")
        print(f"    Shared-word similarity:    {score:.2%}")


def main():
    train_data = load_data(TRAIN_PATH)
    validation_data = load_data(VALIDATION_PATH)
    test_data = load_data(TEST_PATH)

    analyze_dataset(
        "Validation",
        validation_data,
        train_data,
    )

    analyze_dataset(
        "Test",
        test_data,
        train_data,
    )


if __name__ == "__main__":
    main()