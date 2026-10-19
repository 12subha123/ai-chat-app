from pathlib import Path
import sys


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from llm_model.inference.inference_engine import (
    InferenceEngine,
)


# --------------------------------------------------
# Model paths
# --------------------------------------------------

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "tokenizer"
    / "bpe_tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_model"
    / "checkpoints_expanded_320"
    / "checkpoint_epoch_20.pt"
)


# --------------------------------------------------
# Create inference engine
# --------------------------------------------------

_engine = InferenceEngine(
    tokenizer_path=str(
        TOKENIZER_PATH
    ),
    checkpoint_path=str(
        CHECKPOINT_PATH
    ),
)


# --------------------------------------------------
# Generate response
# --------------------------------------------------

def generate_response(
    message: str,
    conversation_context: list | None = None,
) -> str:

    if not message.strip():
        return "Please enter a message."

    response = _engine.generate(
        prompt=message,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    return response