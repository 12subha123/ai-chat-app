from pathlib import Path
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.conversations import router as conversations_router
from app.routes.auth import router as auth_router
from app.routes.voice import router as voice_router

from app.database.db import init_db
from app.core.config import settings
from app.core.logging_config import setup_logging


setup_logging()
init_db()

logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for the AI Chat Application",
    version=settings.APP_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_IMAGES_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "image_generation"
    / "inference"
    / "backend_generated"
)

GENERATED_IMAGES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


app.mount(
    "/generated-images",
    StaticFiles(directory=GENERATED_IMAGES_DIRECTORY),
    name="generated-images",
)



GENERATED_VIDEOS_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "inference"
)

GENERATED_VIDEOS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount(
    "/generated-videos",
    StaticFiles(directory=GENERATED_VIDEOS_DIRECTORY),
    name="generated-videos",
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(auth_router)
app.include_router(voice_router)


@app.get("/")
def root():
    logger.info("Root endpoint was accessed")

    return {
        "message": "AI Chat App backend is running!",
        "status": "success",
    }

