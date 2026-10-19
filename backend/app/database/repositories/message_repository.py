from sqlalchemy.orm import Session

from app.database.models import Message


def get_message_by_id(db: Session, message_id: int):
    return (
        db.query(Message)
        .filter(Message.id == message_id)
        .first()
    )


def get_conversation_messages(
    db: Session,
    conversation_id: int,
):
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
        .all()
    )


def create_message(
    db: Session,
    conversation_id: int,
    content: str,
    role: str,
):
    message = Message(
        conversation_id=conversation_id,
        content=content,
        role=role,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def delete_message(
    db: Session,
    message_id: int,
):
    message = get_message_by_id(db, message_id)

    if message is None:
        return False

    db.delete(message)
    db.commit()

    return True