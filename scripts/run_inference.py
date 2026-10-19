from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from llm_model.inference.inference_engine import InferenceEngine


TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints"
    / "checkpoint_epoch_100.pt"
)


CONTEXT_LENGTH = 128
EMBEDDING_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
FEED_FORWARD_DIM = 1024
DROPOUT = 0.1


def main():

    print("=" * 60)
    print("AI-CHAT-APP INFERENCE")
    print("=" * 60)

    print()
    print("Loading trained model...")

    engine = InferenceEngine(
        tokenizer_path=str(TOKENIZER_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        dropout=DROPOUT,
        bias=True,
    )

    print("Model loaded successfully.")

    print()
    print("Type a question.")
    print("Type 'exit' to quit.")
    print()

    while True:

        prompt = input("You: ").strip()

        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        if not prompt:
            continue

        response = engine.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=1,
            top_p=1.0,
        )

        print()
        print("AI:", response)
        print()


if __name__ == "__main__":
    main()