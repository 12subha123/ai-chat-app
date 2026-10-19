from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm_model.model.transformer import TransformerLanguageModel
from llm_model.tokenizer.bpe_tokenizer import BPETokenizer
from llm_model.inference.sampling import sample_next_token


class InferenceEngine:

    def __init__(
        self,
        tokenizer_path: str,
        checkpoint_path: str,
        context_length: int = 512,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feed_forward_dim: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load tokenizer
        self.tokenizer = BPETokenizer()

        self.tokenizer.load(
            tokenizer_path
        )

        self.vocab_size = len(
            self.tokenizer.token_to_id
        )

        # Create model
        self.model = TransformerLanguageModel(
            vocab_size=self.vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            bias=bias,
        )

        # Load trained checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.context_length = context_length

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:

        if not prompt.strip():
            return ""

        # --------------------------------------------------
        # Build the same conversation format used during
        # training.
        # --------------------------------------------------

        formatted_prompt = (
            "User: "
            + prompt.strip()
            + "\nAssistant:"
        )

        # Tokenize the prompt.
        prompt_token_ids = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False,
        )

        if not prompt_token_ids:
            return ""

        # --------------------------------------------------
        # Keep the complete sequence separately.
        #
        # The model will only SEE the last context_length
        # tokens, but we keep every generated token here.
        # --------------------------------------------------

        all_token_ids = list(
            prompt_token_ids
        )

        generated_token_ids = []

        # Get EOS token ID once.
        eos_token_id = (
            self.tokenizer.token_to_id.get(
                self.tokenizer.EOS_TOKEN
            )
        )

        # --------------------------------------------------
        # Generate new tokens.
        # --------------------------------------------------

        for _ in range(max_new_tokens):

            # Give the model only the most recent tokens
            # that fit inside its context window.
            model_token_ids = (
                all_token_ids[
                    -self.context_length:
                ]
            )

            input_ids = torch.tensor(
                [model_token_ids],
                dtype=torch.long,
                device=self.device,
            )

            # Run the model.
            logits = self.model(
                input_ids
            )

            # We only need predictions for the final
            # position.
            next_token_logits = logits[
                :,
                -1,
                :
            ]

            # Sample the next token.
            next_token_id = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Store the generated token in BOTH places:
            #
            # 1. all_token_ids -> used as model context
            # 2. generated_token_ids -> final answer
            all_token_ids.append(
                next_token_id
            )

            generated_token_ids.append(
                next_token_id
            )

            # --------------------------------------------------
            # Stop at EOS.
            # --------------------------------------------------

            if (
                eos_token_id is not None
                and next_token_id == eos_token_id
            ):
                break

            # --------------------------------------------------
            # Stop if the model starts another User turn.
            # --------------------------------------------------

            generated_text = (
                self.tokenizer.decode(
                    generated_token_ids
                )
            )

            if "user:" in generated_text.lower():
                break

        # --------------------------------------------------
        # Decode ONLY the generated answer.
        # --------------------------------------------------

        response = self.tokenizer.decode(
            generated_token_ids
        )

        # --------------------------------------------------
        # Safety cleanup if another User turn appears.
        # --------------------------------------------------

        lower_response = response.lower()

        user_marker = lower_response.find(
            "user:"
        )

        if user_marker != -1:
            response = response[
                :user_marker
            ]

        return response.strip()


