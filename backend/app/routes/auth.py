from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.database.db import get_db
from app.database.repositories.user_repository import (
    get_user_by_username,
)
from app.schemas.user_schema import UserCreate, UserResponse, LoginRequest, LoginResponse
from app.services.auth_service import register_user


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post("/register", response_model=UserResponse)
def register(
    request: UserCreate,
    db: Session = Depends(get_db),
):
    try:
        user = register_user(
            db=db,
            username=request.username,
            email=request.email,
            password=request.password,
        )

        return user

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.post("/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    user = get_user_by_username(
        db,
        request.username,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    if not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    return {
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
    }