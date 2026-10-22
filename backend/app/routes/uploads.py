from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.ai_core.knowledge.knowledge_service import KnowledgeService


router = APIRouter(
    prefix="/uploads",
    tags=["Uploads"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "app"
    / "uploads"
)

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pdf",
    ".txt",
    ".md",
    ".docx",
}

RAG_EXTENSIONS = {
    ".txt",
    ".md",
}


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    safe_filename = Path(
        file.filename or "uploaded_file"
    ).name

    destination = UPLOAD_DIRECTORY / safe_filename

    contents = await file.read()

    destination.write_bytes(contents)

    ingested_chunks = 0

    if extension in RAG_EXTENSIONS:
        knowledge_service = KnowledgeService(
            knowledge_directory=UPLOAD_DIRECTORY,
            vector_store_directory=PROJECT_ROOT
            / "knowledge"
            / "vector_db",
        )

        ingested_chunks = knowledge_service.ingest_document(
            destination
        )

    return {
        "message": "File uploaded successfully.",
        "filename": safe_filename,
        "file_path": str(destination),
        "content_type": file.content_type,
        "size": len(contents),
        "rag_ingested": extension in RAG_EXTENSIONS,
        "ingested_chunks": ingested_chunks,
    }
