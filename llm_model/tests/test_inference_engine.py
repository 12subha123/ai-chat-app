from pathlib import Path
import sys


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Import
# --------------------------------------------------

from llm_model.inference.inference_engine import (
    InferenceEngine,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

VOCABULARY_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "vocabulary.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints"
    / "checkpoint_epoch_10.pt"
)


# --------------------------------------------------
# Test
# --------------------------------------------------

def main():

    engine = InferenceEngine(
        vocabulary_path=str(
            VOCABULARY_PATH
        ),
        checkpoint_path=str(
            CHECKPOINT_PATH
        ),
    )

    output = engine.generate(
        prompt="Physics is",
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    assert isinstance(
        output,
        str,
    )

    assert len(
        output.strip()
    ) > 0

    print(
        "InferenceEngine loaded successfully!"
    )

    print(
        f"Generated text: {output}"
    )

    print(
        "Inference engine test passed!"
    )


if __name__ == "__main__":
    main()