import json
from pathlib import Path
import sys

import torch


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from llm_model.model.transformer import (
    TransformerLanguageModel,
)

from llm_model.inference.sampling import (
    sample_next_token,
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
# Model configuration
# --------------------------------------------------

CONTEXT_LENGTH = 512
EMBEDDING_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
FEED_FORWARD_DIM = 1024
DROPOUT = 0.1


# --------------------------------------------------
# Generation configuration
# --------------------------------------------------

MAX_NEW_TOKENS = 20
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9


# --------------------------------------------------
# Load vocabulary
# --------------------------------------------------

def load_vocabulary():

    with VOCABULARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    token_to_id = data["token_to_id"]

    id_to_token = {
        int(token_id): token
        for token_id, token in data["id_to_token"].items()
    }

    return token_to_id, id_to_token


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model(
    vocab_size: int,
):

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        dropout=DROPOUT,
        bias=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# --------------------------------------------------
# Encode prompt
# --------------------------------------------------

def encode_prompt(
    prompt: str,
    token_to_id: dict,
) -> list[int]:

    unknown_id = token_to_id["<UNK>"]

    tokens = prompt.split()

    token_ids = [
        token_to_id.get(
            token,
            unknown_id,
        )
        for token in tokens
    ]

    return token_ids


# --------------------------------------------------
# Decode tokens
# --------------------------------------------------

def decode_tokens(
    token_ids: list[int],
    id_to_token: dict,
) -> str:

    tokens = [
        id_to_token.get(
            token_id,
            "<UNK>",
        )
        for token_id in token_ids
    ]

    return " ".join(tokens)


# --------------------------------------------------
# Generate text
# --------------------------------------------------

@torch.no_grad()
def generate(
    model,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[int]:

    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):

        context_ids = generated_ids[
            -CONTEXT_LENGTH:
        ]

        input_tensor = torch.tensor(
            [context_ids],
            dtype=torch.long,
        )

        logits = model(
            input_tensor
        )

        next_token_logits = logits[
            0,
            -1,
            :,
        ]

        next_token_id = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_ids.append(
            next_token_id
        )

    return generated_ids


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AI-CHAT-APP TEXT GENERATION")
    print("=" * 60)

    # Load vocabulary
    token_to_id, id_to_token = (
        load_vocabulary()
    )

    vocab_size = len(token_to_id)

    print(
        f"Vocabulary size: {vocab_size}"
    )

    # Load model
    model = load_model(
        vocab_size=vocab_size
    )

    # Prompt
    prompt = "Physics is"

    input_ids = encode_prompt(
        prompt,
        token_to_id,
    )

    # Generate
    generated_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    # Decode
    output = decode_tokens(
        generated_ids,
        id_to_token,
    )

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Generated text:")
    print(output)

    print()
    print(
        f"Temperature: {TEMPERATURE}"
    )
    print(
        f"Top-k: {TOP_K}"
    )
    print(
        f"Top-p: {TOP_P}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()