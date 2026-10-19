import json
import re
from collections import Counter
from pathlib import Path


class BPETokenizer:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        self.merges = []

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

    # --------------------------------------------------
    # Normalize text
    # --------------------------------------------------

    def normalize(self, text: str) -> str:
        return text.lower().strip()

    # --------------------------------------------------
    # Split text into words
    # --------------------------------------------------

    def _split_words(self, text: str) -> list[str]:
        text = self.normalize(text)

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE,
        )

    # --------------------------------------------------
    # Convert a word into initial BPE symbols
    # --------------------------------------------------

    def _word_to_symbols(
        self,
        word: str,
    ) -> tuple[str, ...]:

        if not word:
            return tuple()

        symbols = []

        for index, character in enumerate(word):

            if index == 0:
                symbols.append(
                    "▁" + character
                )
            else:
                symbols.append(character)

        return tuple(symbols)

    # --------------------------------------------------
    # Count adjacent symbol pairs
    # --------------------------------------------------

    def _get_pair_counts(
        self,
        vocabulary: dict[tuple[str, ...], int],
    ) -> Counter:

        pair_counts = Counter()

        for symbols, frequency in vocabulary.items():

            for index in range(
                len(symbols) - 1
            ):

                pair = (
                    symbols[index],
                    symbols[index + 1],
                )

                pair_counts[pair] += frequency

        return pair_counts

    # --------------------------------------------------
    # Merge one BPE pair
    # --------------------------------------------------

    def _merge_pair(
        self,
        vocabulary: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ):

        new_vocabulary = {}

        first, second = pair

        merged_symbol = first + second

        for symbols, frequency in vocabulary.items():

            new_symbols = []

            index = 0

            while index < len(symbols):

                if (
                    index < len(symbols) - 1
                    and symbols[index] == first
                    and symbols[index + 1] == second
                ):
                    new_symbols.append(
                        merged_symbol
                    )

                    index += 2

                else:
                    new_symbols.append(
                        symbols[index]
                    )

                    index += 1

            new_vocabulary[
                tuple(new_symbols)
            ] = frequency

        return new_vocabulary

    # --------------------------------------------------
    # Build BPE vocabulary
    # --------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int = 1000,
    ) -> None:

        words = self._split_words(text)

        vocabulary = Counter()

        for word in words:

            symbols = self._word_to_symbols(
                word
            )

            vocabulary[symbols] += 1

        self.merges = []

        while True:

            pair_counts = (
                self._get_pair_counts(
                    vocabulary
                )
            )

            if not pair_counts:
                break

            best_pair, frequency = (
                pair_counts.most_common(1)[0]
            )

            if frequency < 2:
                break

            self.merges.append(
                best_pair
            )

            vocabulary = self._merge_pair(
                vocabulary,
                best_pair,
            )

            current_symbols = set()

            for symbols in vocabulary:
                current_symbols.update(
                    symbols
                )

            estimated_vocab_size = (
                len(self.special_tokens)
                + len(current_symbols)
                + len(self.merges)
            )

            if (
                estimated_vocab_size
                >= vocab_size
            ):
                break

        self._build_token_dictionary(
            vocabulary
        )

    # --------------------------------------------------
    # Build token <-> ID dictionaries
    # --------------------------------------------------

    def _build_token_dictionary(
        self,
        vocabulary,
    ):

        tokens = set()

        for symbols in vocabulary:

            for symbol in symbols:
                tokens.add(symbol)

        all_tokens = (
            self.special_tokens
            + sorted(tokens)
        )

        self.token_to_id = {
            token: index
            for index, token
            in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token
            for token, index
            in self.token_to_id.items()
        }

    # --------------------------------------------------
    # Apply learned BPE merges
    # --------------------------------------------------

    def _apply_merges(
        self,
        symbols: tuple[str, ...],
    ) -> list[str]:

        symbols = list(symbols)

        for pair in self.merges:

            first, second = pair

            merged_symbol = (
                first + second
            )

            new_symbols = []

            index = 0

            while index < len(symbols):

                if (
                    index < len(symbols) - 1
                    and symbols[index] == first
                    and symbols[index + 1] == second
                ):
                    new_symbols.append(
                        merged_symbol
                    )

                    index += 2

                else:
                    new_symbols.append(
                        symbols[index]
                    )

                    index += 1

            symbols = new_symbols

        return symbols

    # --------------------------------------------------
    # Encode text
    # --------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:

        words = self._split_words(text)

        token_ids = []

        for word in words:

            symbols = self._word_to_symbols(
                word
            )

            symbols = self._apply_merges(
                symbols
            )

            for symbol in symbols:

                token_id = self.token_to_id.get(
                    symbol,
                    self.token_to_id[
                        self.UNK_TOKEN
                    ],
                )

                token_ids.append(
                    token_id
                )

        if add_special_tokens:

            token_ids = [
                self.token_to_id[
                    self.BOS_TOKEN
                ],
                *token_ids,
                self.token_to_id[
                    self.EOS_TOKEN
                ],
            ]

        return token_ids

    # --------------------------------------------------
    # Decode token IDs
    # --------------------------------------------------

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        tokens = []

        for token_id in token_ids:

            token = self.id_to_token.get(
                token_id,
                self.UNK_TOKEN,
            )

            if token in self.special_tokens:
                continue

            tokens.append(token)

        text = "".join(tokens)

        text = text.replace(
            "▁",
            " ",
        )

        return text.strip()

    # --------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------

    def save(
        self,
        path: str,
    ) -> None:

        tokenizer_path = Path(path)

        tokenizer_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(token_id): token
                for token_id, token
                in self.id_to_token.items()
            },
            "merges": [
                list(pair)
                for pair in self.merges
            ],
        }

        with tokenizer_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    def load(
        self,
        path: str,
    ) -> None:

        tokenizer_path = Path(path)

        with tokenizer_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.token_to_id = (
            data["token_to_id"]
        )

        self.id_to_token = {
            int(token_id): token
            for token_id, token
            in data["id_to_token"].items()
        }

        self.merges = [
            tuple(pair)
            for pair in data["merges"]
        ]