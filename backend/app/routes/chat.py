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


# Create one orchestrator instance for the backend process.
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

    # ---------------------------------------------------------
    # 1. Load existing conversation context
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Build AI Core request
    # ---------------------------------------------------------
    ai_request = AIRequest(
        message=request.message,
        conversation_id=request.conversation_id,
        context=context,
    )

    # ---------------------------------------------------------
    # 3. Let the AI Orchestrator choose the capability
    # ---------------------------------------------------------
    ai_response = _orchestrator.handle(ai_request)

    response = ai_response.message

    # ---------------------------------------------------------
    # 4. Store assistant response in conversation memory
    # ---------------------------------------------------------
    if request.conversation_id is not None:
        add_message(
            db=db,
            conversation_id=request.conversation_id,
            content=response,
            role="assistant",
        )

    # ---------------------------------------------------------
    # 5. Return the same API response format as before
    # ---------------------------------------------------------
    return ChatResponse(
        message=response,
    )