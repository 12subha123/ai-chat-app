from pathlib import Path


def load_token_ids(path: str) -> list[int]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Token data file not found: {file_path}"
        )

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            "Token data file is empty."
        )

    token_ids = [
        int(token)
        for token in text.split()
    ]

    return token_ids