from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.memory_service import get_messages

from app.database.db import get_db
from app.database.repositories.conversation_repository import (
    create_conversation,
    get_user_conversations,
    get_conversation_by_id,
    delete_conversation,
)
from app.schemas.conversation_schema import (
    ConversationCreate,
    ConversationResponse,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


@router.post("", response_model=ConversationResponse)
def create_new_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
):
    conversation = create_conversation(
        db=db,
        user_id=1,
        title=request.title,
    )

    return conversation


@router.get("/user/{user_id}")
def get_conversations(
    user_id: int,
    db: Session = Depends(get_db),
):
    return get_user_conversations(
        db=db,
        user_id=user_id,
    )


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    conversation = get_conversation_by_id(
        db=db,
        conversation_id=conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return conversation


@router.delete("/{conversation_id}")
def remove_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    deleted = delete_conversation(
        db=db,
        conversation_id=conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return {
        "message": "Conversation deleted successfully",
    }

@router.get("/{conversation_id}/messages")
def get_messages_for_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    conversation = get_conversation_by_id(
        db=db,
        conversation_id=conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = get_messages(
        db=db,
        conversation_id=conversation_id,
    )

    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]