from app.routes.chat import router as chat_router 
from app.routes.health import router as health_router
from app.routes.conversations import router as conversations_router
from app.routes.auth import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.database.db import init_db

from fastapi import FastAPI

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(auth_router)

@app.get("/")
def root():
    logger.info("Root endpoint was accessed")

    return {
        "message": "AI Chat App backend is running!",
        "status": "success",
    }


