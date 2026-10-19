import json
from pathlib import Path


class SimpleTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocabulary(self, text: str) -> None:
        tokens = text.split()

        unique_tokens = sorted(set(tokens))

        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        all_tokens = special_tokens + unique_tokens

        self.token_to_id = {
            token: index
            for index, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:

        tokens = text.split()

        unknown_id = self.token_to_id[
            self.UNK_TOKEN
        ]

        token_ids = [
            self.token_to_id.get(
                token,
                unknown_id,
            )
            for token in tokens
        ]

        if add_special_tokens:
            token_ids = [
                self.token_to_id[self.BOS_TOKEN],
                *token_ids,
                self.token_to_id[self.EOS_TOKEN],
            ]

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
            if token_id in self.id_to_token
        ]

        return " ".join(tokens)

    def save_vocabulary(self, path: str) -> None:
        vocabulary_path = Path(path)

        vocabulary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(token_id): token
                for token_id, token in self.id_to_token.items()
            },
        }

        with vocabulary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_vocabulary(self, path: str) -> None:
        vocabulary_path = Path(path)

        with vocabulary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.token_to_id = data["token_to_id"]

        self.id_to_token = {
            int(token_id): token
            for token_id, token in data["id_to_token"].items()
        }