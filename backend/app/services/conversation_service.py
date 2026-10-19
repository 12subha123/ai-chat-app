from sqlalchemy.orm import Session

from app.database.repositories.conversation_repository import (
    create_conversation,
    delete_conversation,
    get_conversation_by_id,
    get_user_conversations,
)


def create_new_conversation(
    db: Session,
    user_id: int,
    title: str,
):
    return create_conversation(
        db=db,
        user_id=user_id,
        title=title,
    )


def get_conversations_for_user(
    db: Session,
    user_id: int,
):
    return get_user_conversations(
        db=db,
        user_id=user_id,
    )


def get_conversation(
    db: Session,
    conversation_id: int,
):
    return get_conversation_by_id(
        db=db,
        conversation_id=conversation_id,
    )


def remove_conversation(
    db: Session,
    conversation_id: int,
):
    return delete_conversation(
        db=db,
        conversation_id=conversation_id,
    )