from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.llm_service import generate_response
from app.services.memory_service import (
    add_message,
    build_conversation_context,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.get("/test")
def chat_test():
    return {
        "message": "Chat route is working!",
        "status": "success",
    }


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    context = []

    if request.conversation_id is not None:
        add_message(
            db=db,
            conversation_id=request.conversation_id,
            content=request.message,
            role="user",
        )

        context = build_conversation_context(
            db=db,
            conversation_id=request.conversation_id,
        )

    response = generate_response(
        message=request.message,
        conversation_context=context,
    )

    if request.conversation_id is not None:
        add_message(
            db=db,
            conversation_id=request.conversation_id,
            content=response,
            role="assistant",
        )

    return ChatResponse(
        message=response,
    )
    