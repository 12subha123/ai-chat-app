from sqlalchemy.orm import Session

from app.database.repositories.message_repository import (
    create_message,
    get_conversation_messages,
)


def add_message(
    db: Session,
    conversation_id: int,
    content: str,
    role: str,
):
    return create_message(
        db=db,
        conversation_id=conversation_id,
        content=content,
        role=role,
    )


def get_messages(
    db: Session,
    conversation_id: int,
):
    return get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
    )


def build_conversation_context(
    db: Session,
    conversation_id: int,
):
    messages = get_messages(
        db=db,
        conversation_id=conversation_id,
    )

    context = []

    for message in messages:
        context.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    return context