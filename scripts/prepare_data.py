from pathlib import Path
import json
import re
import sys


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Input / output paths
# --------------------------------------------------

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "raw"
    / "books.txt"
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

CLEAN_TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train_raw.txt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

TOKENIZED_TRAIN_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "data"
    / "processed"
    / "train.json"
)


# --------------------------------------------------
# Import BPE tokenizer
# --------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.tokenizer.bpe_tokenizer import (
    BPETokenizer,
)


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text: str) -> str:

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# --------------------------------------------------
# Load JSON examples
# --------------------------------------------------

def load_json_examples(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON list."
        )

    for item in data:

        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid example in {path}."
            )

        if (
            "prompt" not in item
            or "answer" not in item
        ):
            raise ValueError(
                f"Each example in {path} "
                "must contain 'prompt' and 'answer'."
            )

    return data


# --------------------------------------------------
# Extract conversation examples
# --------------------------------------------------

def extract_conversations(
    raw_text: str,
) -> list[dict]:

    sections = re.split(
        r"\n\s*\n",
        raw_text,
    )

    conversations = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        user_line = None
        assistant_line = None

        for line in lines:

            stripped = line.strip()

            if stripped.startswith("User:"):
                user_line = stripped

            elif stripped.startswith("Assistant:"):
                assistant_line = stripped

        if user_line is None:
            continue

        if assistant_line is None:
            continue

        user_text = clean_text(
            user_line[len("User:"):].strip()
        )

        assistant_text = clean_text(
            assistant_line[len("Assistant:"):].strip()
        )

        if not user_text:
            continue

        if not assistant_text:
            continue

        conversations.append(
            {
                "prompt": user_text,
                "answer": assistant_text,
            }
        )

    if not conversations:
        raise ValueError(
            "No valid User/Assistant "
            "conversation examples were found."
        )

    return conversations


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("AI-CHAT-APP BPE DATA PREPARATION")
    print("=" * 60)

    # --------------------------------------------------
    # Check raw data
    # --------------------------------------------------

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw training file not found: "
            f"{RAW_DATA_PATH}"
        )

    raw_text = RAW_DATA_PATH.read_text(
        encoding="utf-8"
    )

    raw_text = clean_text(
        raw_text
    )

    if not raw_text:
        raise ValueError(
            "The raw training file is empty."
        )

    # --------------------------------------------------
    # Load validation and test data
    # --------------------------------------------------

    validation_examples = (
        load_json_examples(
            VALIDATION_PATH
        )
    )

    test_examples = (
        load_json_examples(
            TEST_PATH
        )
    )

    # --------------------------------------------------
    # Extract conversations
    # --------------------------------------------------

    conversations = extract_conversations(
        raw_text
    )

    print(
        f"Conversation examples: "
        f"{len(conversations)}"
    )

    # --------------------------------------------------
    # Save cleaned training text
    # --------------------------------------------------

    CLEAN_TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_text = "\n\n".join(
        (
            f"User: {item['prompt']}\n"
            f"Assistant: {item['answer']}"
        )
        for item in conversations
    )

    CLEAN_TRAIN_PATH.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Train BPE tokenizer
    #
    # IMPORTANT:
    # Only training data is used here.
    # Validation and test data remain untouched.
    # --------------------------------------------------

    tokenizer = BPETokenizer()

    tokenizer.train(
        text=cleaned_text,
        vocab_size=1000,
    )

    # --------------------------------------------------
    # Save BPE tokenizer
    # --------------------------------------------------

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(TOKENIZER_PATH)
    )

    # --------------------------------------------------
    # Tokenize each conversation separately
    # --------------------------------------------------

    tokenized_conversations = []

    total_tokens = 0

    for conversation in conversations:

        formatted_conversation = (
            f"User: {conversation['prompt']}\n"
            f"Assistant: {conversation['answer']}"
        )

        conversation_token_ids = (
            tokenizer.encode(
                formatted_conversation,
                add_special_tokens=True,
            )
        )

        tokenized_conversations.append(
            {
                "prompt": conversation["prompt"],
                "answer": conversation["answer"],
                "token_ids": conversation_token_ids,
            }
        )

        total_tokens += len(
            conversation_token_ids
        )

    # --------------------------------------------------
    # Save structured tokenized training data
    # --------------------------------------------------

    TOKENIZED_TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TOKENIZED_TRAIN_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            tokenized_conversations,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print(
        f"Raw training characters: "
        f"{len(raw_text)}"
    )

    print(
        f"Training lines: "
        f"{len(raw_text.splitlines())}"
    )

    print(
        f"BPE vocabulary size: "
        f"{len(tokenizer.token_to_id)}"
    )

    print(
        f"BPE merges: "
        f"{len(tokenizer.merges)}"
    )

    print(
        f"Training tokens: "
        f"{total_tokens}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_examples)}"
    )

    print(
        f"Test examples: "
        f"{len(test_examples)}"
    )

    print(
        f"BPE tokenizer: "
        f"{TOKENIZER_PATH}"
    )

    print(
        f"Structured training data: "
        f"{TOKENIZED_TRAIN_PATH}"
    )

    print("=" * 60)
    print("BPE DATA PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()