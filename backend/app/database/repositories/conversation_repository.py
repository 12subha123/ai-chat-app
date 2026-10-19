from sqlalchemy.orm import Session

from app.database.models import Conversation


def get_conversation_by_id(db: Session, conversation_id: int):
    return (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )


def get_user_conversations(db: Session, user_id: int):
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.id.desc())
        .all()
    )


def create_conversation(
    db: Session,
    user_id: int,
    title: str,
):
    conversation = Conversation(
        user_id=user_id,
        title=title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


def delete_conversation(
    db: Session,
    conversation_id: int,
):
    conversation = get_conversation_by_id(
        db,
        conversation_id,
    )

    if conversation is None:
        return False

    db.delete(conversation)
    db.commit()

    return True