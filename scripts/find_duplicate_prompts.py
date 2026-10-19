import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_merged_320.json"
)


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


prompt_groups = defaultdict(list)


for index, item in enumerate(dataset, start=1):
    prompt = item["prompt"].strip().lower()
    prompt_groups[prompt].append(
        (index, item["answer"].strip())
    )


print("=" * 70)
print("DUPLICATE PROMPT ANSWER COMPARISON")
print("=" * 70)


duplicate_groups = 0


for prompt, examples in prompt_groups.items():

    if len(examples) > 1:

        duplicate_groups += 1

        print()
        print(f"Prompt: {prompt}")
        print("-" * 70)

        for index, answer in examples:

            print(f"Example {index}:")
            print(answer)
            print()


print("=" * 70)
print(
    f"Duplicate prompt groups: {duplicate_groups}"
)
print("=" * 70)