from sqlalchemy.orm import Session

from app.database.repositories.user_repository import (
    create_user,
    get_user_by_email,
    get_user_by_username,
)
from app.core.security import hash_password, verify_password


def register_user(
    db: Session,
    username: str,
    email: str,
    password: str,
):
    existing_username = get_user_by_username(db, username)

    if existing_username:
        raise ValueError("Username already exists")

    existing_email = get_user_by_email(db, email)

    if existing_email:
        raise ValueError("Email already exists")

    password_hash = hash_password(password)

    return create_user(
        db=db,
        username=username,
        email=email,
        password_hash=password_hash,
    )


def authenticate_user(
    db: Session,
    username: str,
    password: str,
):
    user = get_user_by_username(db, username)

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user