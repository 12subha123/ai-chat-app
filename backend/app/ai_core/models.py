from dataclasses import dataclass, field
from typing import Any


@dataclass
class AIRequest:
    message: str
    conversation_id: int | None = None
    user_id: int | None = None
    context: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    message: str
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)