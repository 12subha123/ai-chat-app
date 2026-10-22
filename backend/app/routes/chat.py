from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.memory_service import (
    add_message,
    build_conversation_context,
)
from app.ai_core.models import AIRequest
from app.ai_core.orchestrator import AIOrchestrator
from app.ai_core.knowledge.document_loader import DocumentLoader

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

_orchestrator = AIOrchestrator()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "uploads"
).resolve()


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

    attachment_context = ""

    if request.attachment_path:
        requested_path = Path(
            request.attachment_path
        ).resolve()

        try:
            requested_path.relative_to(
                UPLOAD_DIRECTORY
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid attachment path.",
            )

        if not requested_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Uploaded document not found.",
            )

        if requested_path.suffix.lower() not in {
            ".txt",
            ".md",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This document type cannot be read yet. "
                    "TXT and Markdown files are currently supported."
                ),
            )

        attachment_context = DocumentLoader().load_text(
            requested_path
        )

    ai_message = request.message

    if attachment_context:
        ai_message = (
            "The user has uploaded the following document. "
            "Read the document carefully and use its contents "
            "to answer the user's request.\n\n"
            "----- BEGIN UPLOADED DOCUMENT -----\n"
            f"{attachment_context}\n"
            "----- END UPLOADED DOCUMENT -----\n\n"
            "User request:\n"
            f"{request.message}\n\n"
            "Answer based on the uploaded document."
        )

    ai_request = AIRequest(
        message=ai_message,
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
