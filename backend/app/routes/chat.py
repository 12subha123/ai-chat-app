from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.memory_service import (
    add_message,
    build_conversation_context,
)
from app.ai_core.models import AIRequest
from app.ai_core.orchestrator import AIOrchestrator


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


_orchestrator = AIOrchestrator()


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

    ai_request = AIRequest(
        message=request.message,
        conversation_id=request.conversation_id,
        context=context,
    )

    ai_response = _orchestrator.handle(ai_request)

    response = ai_response.message

    if request.conversation_id is not None:
        add_message(
            db=db,
            conversation_id=request.conversation_id,
            content=response,
            role="assistant",
        )

    image_path = ai_response.metadata.get("image_path")

    image_url = None

    if image_path:
        image_filename = Path(image_path).name
        image_url = (
            f"http://127.0.0.1:8001/generated-images/"
            f"{image_filename}"
        )

    video_path = ai_response.metadata.get("video_path")

    video_url = None

    if video_path:
        video_filename = Path(video_path).name
        video_url = (
            f"http://127.0.0.1:8001/generated-videos/"
            f"{video_filename}"
        )

    return ChatResponse(
        message=response,
        image_path=image_url,
        video_path=video_url,
    )
