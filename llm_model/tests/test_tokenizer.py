from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_model.tokenizer.tokenizer import SimpleTokenizer


def main():
    text = "hello world this is my AI chatbot"

    vocabulary_path = (
        PROJECT_ROOT
        / "llm_model"
        / "tokenizer"
        / "vocabulary.json"
    )

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocabulary(text)

    original_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
    )

    tokenizer.save_vocabulary(
        str(vocabulary_path)
    )

    print("Vocabulary saved to:")
    print(vocabulary_path)

    loaded_tokenizer = SimpleTokenizer()

    loaded_tokenizer.load_vocabulary(
        str(vocabulary_path)
    )

    loaded_ids = loaded_tokenizer.encode(
        text,
        add_special_tokens=True,
    )

    assert original_ids == loaded_ids

    decoded_text = loaded_tokenizer.decode(
        loaded_ids
    )

    print("\nOriginal token IDs:")
    print(original_ids)

    print("\nLoaded token IDs:")
    print(loaded_ids)

    print("\nDecoded text:")
    print(decoded_text)

    assert decoded_text == (
        "<BOS> hello world this is my AI chatbot <EOS>"
    )

    print("\nVocabulary save/load test passed!")


if __name__ == "__main__":
    main()